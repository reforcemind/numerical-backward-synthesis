#include "bw_syn/backends/tt_harness.hpp"
#include "bw_syn/csv_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n\"");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t\r\n\"");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> out;
  std::istringstream in(line);
  std::string field;
  while (std::getline(in, field, ','))
    out.push_back(trim(field));
  return out;
}

std::size_t column(const std::vector<std::string>& names, const std::string& wanted) {
  const auto it = std::find(names.begin(), names.end(), wanted);
  if (it == names.end())
    throw std::runtime_error("profiler CSV lacks column: " + wanted);
  return static_cast<std::size_t>(it - names.begin());
}

using Key = std::tuple<std::string, std::string, std::string, std::string>;
using Interval = std::pair<std::uint64_t, std::uint64_t>;
using Sample = std::pair<std::uint64_t, std::uint64_t>;

// Each TRISC records its own copy of a compute-kernel zone, and slow dispatch leaves
// "run host ID" at 0, so the k-th zone on every TRISC of a core is launch k. A launch
// spans the earliest TRISC start to the latest TRISC end.
std::vector<Sample> launches(const std::string& zone,
                             const std::map<std::string, std::vector<Interval>>& by_risc) {
  std::size_t count = 0;
  for (const auto& [risc, intervals] : by_risc) {
    if (count == 0)
      count = intervals.size();
    else if (intervals.size() != count)
      throw std::runtime_error("TRISC zone counts differ for " + zone);
  }
  std::vector<Sample> out;
  std::uint64_t previous_end = 0;
  for (std::size_t k = 0; k < count; ++k) {
    std::uint64_t start = UINT64_MAX;
    std::uint64_t end = 0;
    for (const auto& [risc, intervals] : by_risc) {
      start = std::min(start, intervals[k].first);
      end = std::max(end, intervals[k].second);
    }
    if (k > 0 && start < previous_end)
      throw std::runtime_error("overlapping launches for " + zone);
    previous_end = end;
    out.emplace_back(start, end - start);
  }
  return out;
}

std::map<std::string, std::vector<Sample>> parse(const std::string& path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open profiler CSV: " + path);
  std::string line;
  std::vector<std::string> header;
  while (std::getline(in, line)) {
    header = fields(line);
    if (std::find(header.begin(), header.end(), "zone name") != header.end())
      break;
  }
  const auto zone_col = column(header, "zone name");
  const auto phase_col = column(header, "type");
  const auto time_col = column(header, "time[cycles since reset]");
  const auto x_col = column(header, "core_x");
  const auto y_col = column(header, "core_y");
  const auto risc_col = column(header, "RISC processor type");
  const auto largest = std::max({zone_col, phase_col, time_col, x_col, y_col, risc_col});

  std::map<Key, std::uint64_t> begins;
  // zone -> core -> RISC -> intervals
  std::map<std::string, std::map<std::string, std::map<std::string, std::vector<Interval>>>>
      intervals;
  while (std::getline(in, line)) {
    const auto row = fields(line);
    if (row.size() <= largest)
      continue;
    const auto& zone = row[zone_col];
    if (zone != "BW_SYN_TANH_FACTORED" && zone != "BW_SYN_TANH_MATERIALIZED")
      continue;
    const auto core = row[x_col] + "-" + row[y_col];
    const Key key{zone, row[x_col], row[y_col], row[risc_col]};
    const auto tick = std::stoull(row[time_col]);
    if (row[phase_col] == "ZONE_START") {
      if (!begins.emplace(key, tick).second)
        throw std::runtime_error("overlapping profiler zone: " + zone);
    } else if (row[phase_col] == "ZONE_END") {
      const auto it = begins.find(key);
      if (it == begins.end() || tick < it->second)
        throw std::runtime_error("unpaired profiler end: " + zone);
      intervals[zone][core][row[risc_col]].emplace_back(it->second, tick);
      begins.erase(it);
    }
  }
  if (!begins.empty())
    throw std::runtime_error("unpaired profiler begin");

  std::map<std::string, std::vector<Sample>> samples;
  for (auto& [zone, cores] : intervals) {
    if (cores.size() != 1)
      throw std::runtime_error("expected one core for " + zone);
    auto& by_risc = cores.begin()->second;
    for (auto& [risc, list] : by_risc)
      std::sort(list.begin(), list.end());
    samples[zone] = launches(zone, by_risc);
  }
  return samples;
}

std::string number(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << value;
  return out.str();
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: summarize_tt_profiler RAW_CSV OUT_CSV ARCH TT_METAL_SHA\n";
    return 2;
  }
  try {
    const std::size_t tiles = bw_syn::tt_harness::critical_cases().size();
    constexpr std::size_t measured = bw_syn::tt_harness::kDeviceMeasuredRuns;
    constexpr std::size_t warmup = bw_syn::tt_harness::kDeviceWarmupRuns;
    const auto samples = parse(argv[1]);
    std::vector<std::string> rows;
    for (const auto& zone : {"BW_SYN_TANH_FACTORED", "BW_SYN_TANH_MATERIALIZED"}) {
      const auto found = samples.find(zone);
      if (found == samples.end() || found->second.size() < measured + warmup + 1)
        throw std::runtime_error(std::string("too few profiler samples for ") + zone);
      auto ordered = found->second;
      std::sort(ordered.begin(), ordered.end());
      std::vector<std::uint64_t> cycles;
      for (auto it = ordered.end() - measured; it != ordered.end(); ++it)
        cycles.push_back(it->second);
      std::sort(cycles.begin(), cycles.end());
      const double median =
          (static_cast<double>(cycles[measured / 2 - 1]) + cycles[measured / 2]) / 2.0;
      const auto p95_index = (95 * measured + 99) / 100 - 1;
      const double p95 = static_cast<double>(cycles[p95_index]);
      rows.push_back(std::string("device,") + argv[3] + "," + argv[4] + "," + zone + "," +
                     std::to_string(tiles) + "," + std::to_string(warmup) + "," +
                     std::to_string(measured) + "," + number(median) + "," + number(p95) + "," +
                     number(median / tiles) + "," + number(p95 / tiles) + "," + argv[1]);
    }
    const std::filesystem::path output(argv[2]);
    if (!output.parent_path().empty())
      std::filesystem::create_directories(output.parent_path());
    if (!bw_syn::write_csv(output.string(),
                           "label,arch,tt_metal_commit,zone,tiles,warmup,measured,median_cycles,"
                           "p95_cycles,median_cycles_per_tile,p95_cycles_per_tile,raw_profiler",
                           rows))
      throw std::runtime_error("cannot write profiler summary");
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
