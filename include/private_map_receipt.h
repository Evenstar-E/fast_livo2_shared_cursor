#pragma once

// Diagnostic-only receipts. This module never grants a public commit.
#include "source_revision_hash.h"
#include <map>
#include <sstream>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace fast_livo::private_map_receipt
{
using Fields = std::map<std::string, std::string>;

inline Fields runtimeHashes()
{
  Fields result;
  result["runtime_sha256_fastlivo_mapping"] =
      source_revision::sha256File("/proc/self/exe");
  const std::set<std::string> wanted{
      "liblaser_mapping.so", "liblio.so", "libvio.so", "libimu_proc.so", "libpre.so"};
  std::map<std::string, std::string> paths;
  std::ifstream maps("/proc/self/maps");
  if (!maps) throw std::runtime_error("cannot inspect loaded project libraries");
  std::string line;
  while (std::getline(maps, line))
  {
    std::istringstream row(line);
    std::string address, permissions, offset, device, inode, path;
    if (!(row >> address >> permissions >> offset >> device >> inode)) continue;
    std::getline(row >> std::ws, path);
    const auto name = std::filesystem::path(path).filename().string();
    if (!wanted.count(name)) continue;
    struct stat info{};
    if (::stat(path.c_str(), &info) != 0 ||
        static_cast<unsigned long long>(info.st_ino) != std::stoull(inode))
      throw std::runtime_error("loaded project library changed on disk");
    if (paths.count(name) && paths[name] != path)
      throw std::runtime_error("ambiguous loaded project library");
    paths[name] = path;
  }
  if (paths.size() != wanted.size())
    throw std::runtime_error("missing loaded project library evidence");
  for (const auto &entry : paths)
    result["runtime_sha256_" + entry.first] = source_revision::sha256File(entry.second);
  return result;
}

inline void writeAtomic(const std::filesystem::path &receipt, Fields fields,
                        const std::filesystem::path &map_path)
{
  // Caller owns a newly created snapshot directory. Never overwrite old evidence.
  const auto temporary = receipt.string() + ".tmp";
  if (std::filesystem::exists(receipt) || std::filesystem::exists(temporary))
    throw std::runtime_error("private map receipt already exists");
  fields["schema"] = "private_map_receipt_v1";
  fields["role"] = "candidate_private";
  fields["phase"] = "replay_complete_pre_validation";
  fields["evaluation_only"] = "1";
  fields["public_commit"] = "0";
  fields["map_file"] = map_path.filename().string();
  fields["map_sha256"] = source_revision::sha256File(map_path);
  for (const auto &entry : fields)
    if (entry.first.empty() || entry.first.find_first_of("=\r\n") != std::string::npos ||
        entry.second.find_first_of("\r\n") != std::string::npos)
      throw std::runtime_error("invalid private map receipt field");
  std::ofstream output(temporary, std::ios::out | std::ios::trunc);
  for (const auto &entry : fields) output << entry.first << '=' << entry.second << '\n';
  output.flush();
  if (!output) throw std::runtime_error("private map receipt write failed");
  output.close();
  std::filesystem::rename(temporary, receipt);
}
}  // namespace fast_livo::private_map_receipt
