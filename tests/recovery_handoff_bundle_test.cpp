#include "LIVMapper.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

int main()
{
  std::deque<LIVMapper::BufferedTransactionEvent> replay_events(3);
  replay_events[0].relative_time = 42.30;
  replay_events[1].relative_time = 42.35;
  replay_events[2].relative_time = 42.40;
  const auto continuous = LIVMapper::assessRecoveryReplayCoverage(
      42.25, replay_events, 0.25);
  assert(continuous.continuous);
  assert(std::abs(continuous.gap_sec - 0.05) < 1e-9);
  const auto missing_prefix = LIVMapper::assessRecoveryReplayCoverage(
      41.0, replay_events, 0.25);
  assert(!missing_prefix.continuous);
  assert(std::abs(missing_prefix.gap_sec - 1.30) < 1e-9);
  replay_events[2].relative_time = 42.34;
  const auto nonmonotonic = LIVMapper::assessRecoveryReplayCoverage(
      43.0, replay_events, 0.25);
  assert(!nonmonotonic.continuous);

  LIVMapper::RecoveryHandoffBundle bundle;
  bundle.generation = 17;
  bundle.boundary_relative_time = 42.25;
  bundle.next_event_sequence = 81;
  bundle.visual_patch_size_total = 4;
  bundle.mapper.state.pos_end = V3D(1.0, 2.0, 3.0);
  bundle.mapper.lidar_map_inited = true;
  bundle.detector.baseline.push_back(
      {40.0, 0.02, 0.9, 0.1, false});
  bundle.detector.long_history.push_back(
      {41.0, 0.03, 0.8, 0.2, true});
  bundle.detector.sample_intervals_sec.push_back(0.1);
  bundle.detector.episode_id = 3;
  bundle.replay_journal_path = "/tmp/replay.bin";
  fast_livo::replay_storage::Record record;
  record.offset = 100;
  record.size = 200;
  record.checksum = 300;
  record.spilled = true;
  bundle.replay_records.push_back(record);

  VoxelMapConfig config{};
  config.max_voxel_size_ = 1.0;
  config.max_layer_ = 2;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> initial_map;
  VoxelMapManager voxel_manager(config, initial_map);
  VoxelOctoTree *root = new VoxelOctoTree(2, 0, 5, 50, 0.01f);
  root->plane_ptr_->center_ = V3D(4.0, 5.0, 6.0);
  voxel_manager.voxel_map_.emplace(VOXEL_LOCATION(1, 2, 3), root);
  bundle.voxel_map = voxel_manager.CreateSnapshot();

  bundle.visual_map.reset(new VIOCheckpoint());
  VOXEL_POINTS *bucket = new VOXEL_POINTS(1);
  VisualPoint *visual_point = new VisualPoint(V3D(0.5, 1.0, 1.5));
  float *patch = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
  Feature *feature = new Feature(visual_point, patch, V2D(2.0, 3.0),
                                 V3D(0.0, 0.0, 1.0), SE3(), 0);
  visual_point->obs_.push_back(feature);
  visual_point->ref_patch = feature;
  visual_point->has_ref_patch_ = true;
  bucket->voxel_points.push_back(visual_point);
  bundle.visual_map->feat_map.emplace(VOXEL_LOCATION(1, 2, 3), bucket);

  const std::vector<std::uint8_t> encoded =
      LIVMapper::serializeRecoveryHandoffBundle(bundle);
  LIVMapper::RecoveryHandoffBundle decoded =
      LIVMapper::deserializeRecoveryHandoffBundle(encoded);
  LIVMapper::RecoveryHandoffBundle decoded_view =
      LIVMapper::deserializeRecoveryHandoffBundle(
          encoded.data(), encoded.size());
  assert(decoded.generation == 17);
  assert(decoded.boundary_relative_time == 42.25);
  assert(decoded.next_event_sequence == 81);
  assert(decoded.detector.baseline.front().match_ratio == 0.9);
  assert(decoded.detector.episode_id == 3);
  assert(decoded.replay_records.front().checksum == 300);
  assert(decoded.voxel_map->fingerprint() == bundle.voxel_map->fingerprint());
  assert(decoded_view.voxel_map->fingerprint() ==
         bundle.voxel_map->fingerprint());
  assert(decoded_view.visual_map->feat_map.size() == 1);
  assert(LIVMapper::serializeRecoveryHandoffBundle(decoded) == encoded);

  std::vector<std::uint8_t> corrupted = encoded;
  corrupted[encoded.size() / 2] ^= 0x04;
  bool rejected = false;
  try
  {
    LIVMapper::deserializeRecoveryHandoffBundle(corrupted);
  }
  catch (const std::runtime_error &)
  {
    rejected = true;
  }
  assert(rejected);

  const std::string path = "/tmp/fast_livo_handoff_" +
      std::to_string(static_cast<long long>(::getpid())) + ".bin";
  LIVMapper::writeRecoveryHandoffBundleAtomic(path, bundle);
  const LIVMapper::RecoveryHandoffHeader header =
      LIVMapper::readRecoveryHandoffHeader(path);
  assert(header.generation == 17);
  assert(header.boundary_relative_time == 42.25);
  assert(header.next_event_sequence == 81);
  assert(header.visual_patch_size_total == 4);
  LIVMapper::RecoveryHandoffBundle from_disk =
      LIVMapper::readRecoveryHandoffBundle(path);
  assert(from_disk.generation == 17);
  assert(from_disk.mapper.state.pos_end == V3D(1.0, 2.0, 3.0));
  std::remove(path.c_str());

  const std::string journal = "/tmp/fast_livo_fork_journal_" +
      std::to_string(static_cast<long long>(::getpid()));
  {
    std::ofstream stream(journal.c_str(), std::ios::binary);
    stream << "sealed-prefix";
  }
  ::chmod(journal.c_str(), 0644);
  assert(LIVMapper::authenticatePrivateForkJournalFiles(
      journal, journal, "/tmp/other_spill").find("writable") !=
      std::string::npos);
  ::chmod(journal.c_str(), 0444);
  assert(LIVMapper::authenticatePrivateForkJournalFiles(
      journal, journal, "/tmp/other_spill").empty());
  assert(LIVMapper::authenticatePrivateForkJournalFiles(
      journal, journal, journal).find("writable_spill") != std::string::npos);
  assert(LIVMapper::authenticatePrivateForkJournalFiles(
      "", journal, "/tmp/other_spill").find("missing") != std::string::npos);
  assert(LIVMapper::authenticatePrivateForkJournalFiles(
      journal, journal + ".other", "/tmp/other_spill").find("mismatch") !=
      std::string::npos);
  std::remove(journal.c_str());

  voxel_manager.ClearVoxelMap();
  std::cout << "RECOVERY_HANDOFF_BUNDLE_TEST_OK" << std::endl;
  return 0;
}
