#include "vio.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main()
{
  VIOManager manager;
  manager.patch_size_total = 4;
  manager.length = 3;
  manager.frame_count = 17;
  manager.ave_total = 0.012;
  manager.visual_submap = new SubSparseMap();

  const VOXEL_LOCATION key(1, 2, 3);
  VOXEL_POINTS *voxel = new VOXEL_POINTS(1);
  // Coordinates must agree with the production 0.5 m visual-voxel index.
  VisualPoint *point = new VisualPoint(V3D(0.5, 1.0, 1.5));
  point->normal_ = V3D(0.0, 0.0, 1.0);
  point->is_normal_initialized_ = true;
  float *patch = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
  Feature *feature = new Feature(point, patch, V2D(10.0, 20.0), V3D(0.0, 0.0, 1.0), SE3(), 0);
  feature->id_ = 9;
  feature->mean_ = 2.5f;
  feature->score_ = 0.25f;
  feature->img_ = (cv::Mat_<std::uint8_t>(2, 3) << 1, 2, 3, 4, 5, 6);
  point->obs_.push_back(feature);
  point->ref_patch = feature;
  point->has_ref_patch_ = true;
  voxel->voxel_points.push_back(point);
  manager.feat_map.emplace(key, voxel);

  const auto checkpoint = manager.createCheckpoint();
  assert(checkpoint->feat_map.at(key) != manager.feat_map.at(key));
  assert(checkpoint->feat_map.at(key)->voxel_points.front() != point);

  // Persist the complete landmark graph, including patch pixels and the
  // reference-feature identity, through the cross-process byte format.
  const std::vector<std::uint8_t> encoded =
      manager.serializeCheckpoint(*checkpoint);
  const std::uint64_t fingerprint =
      manager.checkpointFingerprint(*checkpoint);
  const auto decoded = manager.deserializeCheckpoint(encoded);
  assert(manager.serializeCheckpoint(*decoded) == encoded);
  assert(manager.checkpointFingerprint(*decoded) == fingerprint);
  VisualPoint *decoded_point = decoded->feat_map.at(key)->voxel_points.front();
  assert(decoded_point->ref_patch != nullptr);
  assert(decoded_point->ref_patch->point_ == decoded_point);
  assert(decoded_point->ref_patch->img_.at<std::uint8_t>(1, 2) == 6);
  assert(decoded_point->ref_patch->patch_[3] == 4.0f);
  std::vector<std::uint8_t> corrupted = encoded;
  corrupted[corrupted.size() / 2] ^= 0x80;
  bool corruption_rejected = false;
  try
  {
    manager.deserializeCheckpoint(corrupted);
  }
  catch (const std::runtime_error &)
  {
    corruption_rejected = true;
  }
  assert(corruption_rejected);

  point->pos_.x() = 99.0;
  feature->patch_[0] = 99.0f;
  feature->mean_ = 99.0f;
  manager.frame_count = 99;
  manager.feat_map.emplace(VOXEL_LOCATION(8, 8, 8), new VOXEL_POINTS(0));

  manager.restoreCheckpoint(*checkpoint);
  assert(manager.feat_map.size() == 1);
  VisualPoint *restored = manager.feat_map.at(key)->voxel_points.front();
  assert(std::abs(restored->pos_.x() - 0.5) < 1e-12);
  assert(restored->ref_patch != nullptr);
  assert(restored->ref_patch->point_ == restored);
  assert(std::abs(restored->ref_patch->patch_[0] - 1.0f) < 1e-6f);
  assert(std::abs(restored->ref_patch->mean_ - 2.5f) < 1e-6f);
  assert(manager.frame_count == 17);
  assert(std::abs(manager.ave_total - 0.012) < 1e-12);
  assert(manager.retrieve_voxel_points.size() == 3);
  assert(manager.visual_submap->voxel_points.empty());

  // The rolling path keeps only metadata plus first-write bucket images.
  const auto metadata = manager.createMetadataCheckpoint();
  const std::size_t older_journal = manager.beginMutationJournal();
  VisualPoint *baseline = manager.feat_map.at(key)->voxel_points.front();
  manager.recordPointMutation(*baseline);
  baseline->normal_.x() = 7.0;
  baseline->ref_patch->patch_[0] = 8.0f;
  assert(manager.mutationJournalBeforeImageCount(older_journal) == 1);
  VisualPoint *appended = new VisualPoint(V3D(0.5, 1.0, 1.5));
  manager.insertPointIntoVoxelMap(appended);
  assert(manager.mutationJournalBeforeImageCount(older_journal) == 1);
  assert(manager.mutationJournalInsertedCount(older_journal) == 1);
  assert(manager.feat_map.at(key)->voxel_points.size() == 2);

  const std::size_t newer_journal = manager.beginMutationJournal();
  VisualPoint *inserted = new VisualPoint(V3D(4.0, 4.0, 4.0));
  manager.insertPointIntoVoxelMap(inserted);
  assert(manager.mutationJournalInsertedCount(older_journal) == 2);
  assert(manager.mutationJournalInsertedCount(newer_journal) == 1);
  manager.discardMutationJournal(newer_journal);
  assert(!manager.mutationJournalActive(newer_journal));

  manager.frame_count = 42;
  manager.rollbackMutationJournal(older_journal);
  manager.restoreCheckpointMetadata(*metadata);
  assert(manager.feat_map.size() == 1);
  assert(manager.feat_map.at(key)->voxel_points.size() == 1);
  assert(std::abs(manager.feat_map.at(key)->voxel_points.front()->normal_.x()) < 1e-12);
  assert(std::abs(manager.feat_map.at(key)->voxel_points.front()->ref_patch->patch_[0] - 1.0f) < 1e-6f);
  assert(manager.frame_count == 17);
  assert(!manager.mutationJournalActive());

  // Capture only dirty visual buckets as an exact boundary after-image, then
  // prove it survives an intervening speculative generation.
  const std::size_t redo_journal = manager.beginMutationJournal();
  VisualPoint *boundary_point = manager.feat_map.at(key)->voxel_points.front();
  manager.recordPointMutation(*boundary_point);
  boundary_point->normal_.x() = 3.0;
  VisualPoint *boundary_inserted = new VisualPoint(V3D(0.5, 1.0, 1.5));
  manager.insertPointIntoVoxelMap(boundary_inserted);
  auto redo = manager.captureMutationRedo(redo_journal);
  assert(redo);
  assert(redo->bucket_after_images.size() == 1);
  manager.rollbackMutationJournal(redo_journal);
  const std::size_t speculative = manager.beginMutationJournal(redo.get());
  VisualPoint *speculative_point =
      manager.feat_map.at(key)->voxel_points.front();
  manager.recordPointMutation(*speculative_point);
  speculative_point->normal_.x() = 11.0;
  assert(manager.mutationJournalBeforeImageCount(speculative) == 0);
  manager.rollbackMutationJournal(speculative);
  assert(manager.consumeMutationRedo(*redo));
  assert(manager.feat_map.at(key)->voxel_points.size() == 2);
  assert(std::abs(manager.feat_map.at(key)->voxel_points.front()->normal_.x() -
                  3.0) < 1e-12);

  std::cout << "VIO_CHECKPOINT_TEST_OK" << std::endl;
  return 0;
}
