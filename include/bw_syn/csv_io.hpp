#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace bw_syn {

inline bool write_csv(const std::string& path,
                      const std::string& header,
                      const std::vector<std::string>& rows) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs)
    return false;
  ofs << header << "\n";
  for (const auto& r : rows)
    ofs << r << "\n";
  return true;
}

inline std::string hex16(std::uint16_t b) {
  static const char* k = "0123456789abcdef";
  std::string s = "0x";
  s.push_back(k[(b >> 12) & 0xf]);
  s.push_back(k[(b >> 8) & 0xf]);
  s.push_back(k[(b >> 4) & 0xf]);
  s.push_back(k[b & 0xf]);
  return s;
}

inline std::string csv_f32(float x) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.9g", static_cast<double>(x));
  return b;
}

} // namespace bw_syn
