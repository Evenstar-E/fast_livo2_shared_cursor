#pragma once

// Default-off map captured at one typed event, before rollback.
// A relative time alone is not an identity: LIO and VIO may share it.
#include "source_revision_hash.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fast_livo::typed_event_map
{
using Fields = std::map<std::string, std::string>;

inline double completedTimeFromBoundaryLabel(double label)
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &label, sizeof(bits));
  ++bits;
  double completed = 0.0;
  std::memcpy(&completed, &bits, sizeof(completed));
  return completed;
}

inline bool typedEventMatches(const std::string &observed_type,
                              double observed_relative_sec,
                              const std::string &expected_type,
                              double expected_relative_sec)
{
  if (observed_type != "LIO" && observed_type != "VIO") return false;
  if (observed_type != expected_type) return false;
  if (!std::isfinite(observed_relative_sec) ||
      !std::isfinite(expected_relative_sec))
    return false;
  return std::abs(observed_relative_sec - expected_relative_sec) <= 2e-6;
}

inline void refuseRollbackImpersonation(const Fields &fields)
{
  const auto role = fields.find("branch_role");
  if (role == fields.end() ||
      (role->second != "direct" && role->second != "identity_replay"))
    throw std::runtime_error("typed snapshot role is not a live branch");
  const auto event_type = fields.find("event_type");
  if (event_type == fields.end() ||
      (event_type->second != "LIO" && event_type->second != "VIO"))
    throw std::runtime_error("typed snapshot requires LIO or VIO");
  const auto rollback = fields.find("rollback_state");
  if (rollback != fields.end() && rollback->second != "0")
    throw std::runtime_error("rollback map cannot use a typed-event snapshot");
  const auto phase = fields.find("phase");
  if (phase != fields.end() && phase->second != "before_rollback")
    throw std::runtime_error("typed snapshot phase is after rollback");
  const auto stage = fields.find("processing_stage");
  if (stage != fields.end() && stage->second != "post_update")
    throw std::runtime_error("typed snapshot stage is not post_update");
  const auto file = fields.find("map_file");
  if (file != fields.end() &&
      (file->second.find("rejected") != std::string::npos ||
       file->second.find("rollback") != std::string::npos))
    throw std::runtime_error("rejected or rollback file cannot be the snapshot");
}

inline void writeAtomic(const std::filesystem::path &receipt, Fields fields,
                        const std::filesystem::path &map_path)
{
  refuseRollbackImpersonation(fields);
  const auto temporary = receipt.string() + ".tmp";
  if (std::filesystem::exists(receipt) || std::filesystem::exists(temporary))
    throw std::runtime_error("typed event map receipt already exists");
  fields["schema"] = "typed_event_map_snapshot_v1";
  fields["phase"] = "before_rollback";
  fields["processing_stage"] = "post_update";
  fields["rollback_state"] = "0";
  fields["evaluation_only"] = "1";
  fields["public_commit"] = "0";
  fields["reference_frame"] = "camera_init";
  fields["export_policy"] = "bounded_voxel_surfel_v1";
  fields["map_file"] = map_path.filename().string();
  if (fields["map_file"] != "typed_event_map.pcd")
    throw std::runtime_error("typed snapshot map filename is fixed");
  fields["map_sha256"] = source_revision::sha256File(map_path);
  for (const auto &entry : fields)
    if (entry.first.empty() ||
        entry.first.find_first_of("=\r\n") != std::string::npos ||
        entry.second.find_first_of("\r\n") != std::string::npos)
      throw std::runtime_error("invalid typed event map receipt field");
  std::ofstream output(temporary, std::ios::out | std::ios::trunc);
  for (const auto &entry : fields)
    output << entry.first << '=' << entry.second << '\n';
  output.flush();
  if (!output) throw std::runtime_error("typed event map receipt write failed");
  output.close();
  std::filesystem::rename(temporary, receipt);
}
}  // namespace fast_livo::typed_event_map
