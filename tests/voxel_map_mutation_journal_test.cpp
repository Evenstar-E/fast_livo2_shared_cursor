#include "voxel_map.h"

#include <cassert>
#include <iostream>

namespace
{
pointWithVar makePoint(const double x, const double y, const double z)
{
  pointWithVar point;
  point.point_w = Eigen::Vector3d(x, y, z);
  point.var = Eigen::Matrix3d::Identity() * 1e-4;
  return point;
}

VoxelOctoTree *makeRoot(const Eigen::Vector3d &center)
{
  VoxelOctoTree *root = new VoxelOctoTree(2, 0, 5, 50, 0.01f);
  root->layer_init_num_ = {5, 5, 5};
  root->quater_length_ = 0.25f;
  for (int axis = 0; axis < 3; ++axis) root->voxel_center_[axis] = center[axis];
  root->plane_ptr_->center_ = center;
  return root;
}
} // namespace

int main()
{
  VoxelMapConfig config{};
  config.max_voxel_size_ = 1.0;
  config.max_layer_ = 2;
  config.layer_init_num_ = {5, 5, 5};
  config.max_points_num_ = 50;
  config.planner_threshold_ = 0.01;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> initial_map;
  VoxelMapManager manager(config, initial_map);

  const VOXEL_LOCATION verified_key(1, 2, 3);
  manager.voxel_map_.emplace(verified_key,
                             makeRoot(Eigen::Vector3d(1.5, 2.5, 3.5)));
  VoxelOctoTree *first_generation_root = manager.voxel_map_.at(verified_key);

  // Existing roots receive one before-image regardless of repeat writes; new
  // roots retain only their keys and disappear on rollback.
  manager.BeginMutationJournal();
  manager.UpdateVoxelMap({makePoint(1.25, 2.25, 3.25),
                          makePoint(1.35, 2.35, 3.35),
                          makePoint(9.25, 9.25, 9.25)});
  assert(manager.MutationJournalBeforeImageCount() == 1);
  assert(manager.MutationJournalInsertedCount() == 1);
  assert(manager.MutationJournalDirtyCount() == 2);
  assert(manager.MutationJournalPeakDirtyCount() == 2);
  assert(manager.voxel_map_.size() == 2);
  manager.RollbackMutationJournal();
  assert(!manager.MutationJournalActive());
  assert(manager.voxel_map_.size() == 1);
  assert(manager.voxel_map_.find(VOXEL_LOCATION(9, 9, 9)) ==
         manager.voxel_map_.end());
  assert(manager.voxel_map_.at(verified_key) != first_generation_root);
  assert(manager.voxel_map_.at(verified_key)->temp_points_.empty());

  // Sliding deletion is reversible and produces a fresh before-image in the
  // next generation.
  VoxelOctoTree *second_generation_root = manager.voxel_map_.at(verified_key);
  manager.BeginMutationJournal();
  manager.clearMemOutOfMap(0, 0, 0, 0, 0, 0);
  assert(manager.voxel_map_.empty());
  assert(manager.MutationJournalBeforeImageCount() == 1);
  manager.RollbackMutationJournal();
  assert(manager.voxel_map_.size() == 1);
  assert(manager.voxel_map_.at(verified_key) != second_generation_root);
  assert(manager.voxel_map_.at(verified_key)->plane_ptr_->center_.isApprox(
      Eigen::Vector3d(1.5, 2.5, 3.5)));

  // Two overlapping generations retain different before-images. Selecting
  // the older anchor must restore its state and retire every newer journal.
  const std::size_t older_journal = manager.BeginMutationJournal();
  manager.UpdateVoxelMap({makePoint(1.35, 2.35, 3.35)});
  const std::size_t newer_journal = manager.BeginMutationJournal();
  manager.UpdateVoxelMap({makePoint(1.45, 2.45, 3.45)});
  assert(manager.MutationJournalActive(older_journal));
  assert(manager.MutationJournalActive(newer_journal));
  assert(manager.MutationJournalBeforeImageCount(older_journal) == 1);
  assert(manager.MutationJournalBeforeImageCount(newer_journal) == 1);
  manager.DiscardMutationJournal(newer_journal);
  assert(!manager.MutationJournalActive(newer_journal));
  assert(manager.MutationJournalActive(older_journal));
  manager.RollbackMutationJournal(older_journal);
  assert(!manager.MutationJournalActive());
  assert(manager.voxel_map_.at(verified_key)->temp_points_.empty());

  // Commit deliberately preserves both an in-place update and a new root.
  manager.BeginMutationJournal();
  manager.UpdateVoxelMap({makePoint(1.45, 2.45, 3.45),
                          makePoint(7.25, 8.25, 9.25)});
  manager.CommitMutationJournal();
  assert(!manager.MutationJournalActive());
  assert(manager.voxel_map_.size() == 2);
  assert(manager.voxel_map_.at(verified_key)->temp_points_.size() == 1);
  assert(manager.voxel_map_.find(VOXEL_LOCATION(7, 8, 9)) !=
         manager.voxel_map_.end());

  // Updating a mature frozen plane is a semantic no-op and must not allocate
  // a deep before-image in every overlapping rolling journal.
  VoxelOctoTree *frozen_root = manager.voxel_map_.at(verified_key);
  frozen_root->init_octo_ = true;
  frozen_root->plane_ptr_->is_plane_ = true;
  frozen_root->update_enable_ = false;
  const std::size_t frozen_points = frozen_root->temp_points_.size();
  manager.BeginMutationJournal();
  manager.UpdateVoxelMap({makePoint(1.45, 2.45, 3.45)});
  assert(manager.MutationJournalBeforeImageCount() == 0);
  assert(manager.MutationJournalInsertedCount() == 0);
  assert(manager.MutationJournalDirtyCount() == 0);
  assert(manager.voxel_map_.at(verified_key) == frozen_root);
  assert(frozen_root->temp_points_.size() == frozen_points);
  manager.RollbackMutationJournal();
  assert(manager.voxel_map_.at(verified_key) == frozen_root);

  // A mutable leaf below an initialized branch is journaled at leaf granularity:
  // rollback replaces the leaf but keeps the much larger root allocation.
  const VOXEL_LOCATION branched_key(20, 20, 20);
  VoxelOctoTree *branched_root = makeRoot(Eigen::Vector3d(20.5, 20.5, 20.5));
  branched_root->init_octo_ = true;
  branched_root->plane_ptr_->is_plane_ = false;
  branched_root->octo_state_ = 1;
  branched_root->leaves_[0] = new VoxelOctoTree(2, 1, 5, 50, 0.01f);
  branched_root->leaves_[0]->layer_init_num_ = {5, 5, 5};
  branched_root->leaves_[0]->voxel_center_[0] = 20.25;
  branched_root->leaves_[0]->voxel_center_[1] = 20.25;
  branched_root->leaves_[0]->voxel_center_[2] = 20.25;
  branched_root->leaves_[0]->quater_length_ = 0.125f;
  manager.voxel_map_.emplace(branched_key, branched_root);
  VoxelOctoTree *original_leaf = branched_root->leaves_[0];
  manager.BeginMutationJournal();
  manager.UpdateVoxelMap({makePoint(20.25, 20.25, 20.25)});
  assert(manager.MutationJournalBeforeImageCount() == 1);
  assert(manager.voxel_map_.at(branched_key) == branched_root);
  assert(branched_root->leaves_[0] == original_leaf);
  assert(original_leaf->temp_points_.size() == 1);
  manager.RollbackMutationJournal();
  assert(manager.voxel_map_.at(branched_key) == branched_root);
  assert(branched_root->leaves_[0] != original_leaf);
  assert(branched_root->leaves_[0]->temp_points_.empty());

  // A compact redo image must restore the exact boundary generation after a
  // speculative mutation is rolled back to the anchor.
  const std::size_t redo_journal = manager.BeginMutationJournal();
  manager.UpdateVoxelMap({makePoint(20.25, 20.25, 20.25),
                          makePoint(30.25, 30.25, 30.25)});
  const std::size_t boundary_leaf_points =
      manager.voxel_map_.at(branched_key)->leaves_[0]->temp_points_.size();
  const std::size_t boundary_size = manager.voxel_map_.size();
  auto redo = manager.CaptureMutationRedo(redo_journal);
  assert(redo);
  assert(!redo->node_after_images.empty());
  manager.RollbackMutationJournal(redo_journal);
  assert(manager.voxel_map_.size() + 1 == boundary_size);
  const std::size_t speculative = manager.BeginMutationJournal(redo.get());
  manager.UpdateVoxelMap({makePoint(20.35, 20.35, 20.35)});
  assert(manager.MutationJournalDirtyCount(speculative) == 0);
  manager.RollbackMutationJournal(speculative);
  assert(manager.ConsumeMutationRedo(*redo));
  assert(manager.voxel_map_.size() == boundary_size);
  assert(manager.voxel_map_.find(VOXEL_LOCATION(30, 30, 30)) !=
         manager.voxel_map_.end());
  assert(manager.voxel_map_.at(branched_key)->leaves_[0]->temp_points_.size() ==
         boundary_leaf_points);

  manager.ClearVoxelMap();
  std::cout << "VOXEL_MAP_MUTATION_JOURNAL_TEST_OK" << std::endl;
  return 0;
}
