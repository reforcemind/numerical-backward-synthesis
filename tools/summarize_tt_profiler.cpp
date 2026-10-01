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

using Key = std::tuple<std::string, std::string, std::string, std::string, std::string>;
using Sample = std::pair<std::uint64_t, std::uint64_t>;

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
  const auto phase_col = column(header, "zone phase");
  const auto time_col = column(header, "time[cycles since reset]");
  const auto run_col = column(header, "Run ID");
  const auto x_col = column(header, "core_x");
  const auto y_col = column(header, "core_y");
  const auto risc_col = column(header, "RISC processor type");
  const auto largest = std::max({zone_col, phase_col, time_col, run_col, x_col, y_col, risc_col});

  std::map<Key, std::uint64_t> begins;
  std::map<std::string, std::vector<Sample>> samples;
  while (std::getline(in, line)) {
    const auto row = fields(line);
    if (row.size() <= largest)
      continue;
    const auto& zone = row[zone_col];
    if (zone != "BW_SYN_TANH_FACTORED" && zone != "BW_SYN_TANH_MATERIALIZED")
      continue;
    const Key key{zone, row[run_col], row[x_col], row[y_col], row[risc_col]};
    const auto tick = std::stoull(row[time_col]);
    if (row[phase_col] == "begin") {
      if (!begins.emplace(key, tick).second)
        throw std::runtime_error("overlapping profiler zone: " + zone);
    } else if (row[phase_col] == "end") {
      const auto it = begins.find(key);
      if (it == begins.end() || tick < it->second)
        throw std::runtime_error("unpaired profiler end: " + zone);
      samples[zone].emplace_back(it->second, tick - it->second);
      begins.erase(it);
    }
  }
  if (!begins.empty())
    throw std::runtime_error("unpaired profiler begin");
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
