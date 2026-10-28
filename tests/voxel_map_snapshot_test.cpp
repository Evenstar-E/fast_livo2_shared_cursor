#include "voxel_map.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main()
{
  VoxelMapConfig config{};
  config.max_voxel_size_ = 1.0;
  config.max_layer_ = 2;
  config.sigma_num_ = 10.0;
  config.beam_err_ = 0.01;
  config.dept_err_ = 0.01;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> initial_map;
  VoxelMapManager manager(config, initial_map);

  const VOXEL_LOCATION verified_location(1, 2, 3);
  VoxelOctoTree *verified_root = new VoxelOctoTree(2, 0, 5, 50, 0.01f);
  verified_root->voxel_center_[0] = 1.0;
  verified_root->voxel_center_[1] = 2.0;
  verified_root->voxel_center_[2] = 3.0;
  verified_root->plane_ptr_->center_ = Eigen::Vector3d(1.0, 2.0, 3.0);
  verified_root->leaves_[0] = new VoxelOctoTree(2, 1, 5, 50, 0.01f);
  verified_root->leaves_[0]->plane_ptr_->points_size_ = 7;
  manager.voxel_map_.emplace(verified_location, verified_root);

  std::unique_ptr<VoxelMapSnapshot> snapshot = manager.CreateSnapshot();
  assert(snapshot->size() == 1);

  // Cross-process handoff uses a deterministic, pointer-free byte stream.
  // A decoded generation must be bit-identical when encoded again, and a
  // damaged payload must never become a candidate seed.
  const std::vector<std::uint8_t> encoded_snapshot = snapshot->serialize();
  const std::uint64_t snapshot_fingerprint = snapshot->fingerprint();
  std::unique_ptr<VoxelMapSnapshot> decoded_snapshot =
      VoxelMapSnapshot::deserialize(encoded_snapshot);
  assert(decoded_snapshot->size() == snapshot->size());
  assert(decoded_snapshot->serialize() == encoded_snapshot);
  assert(decoded_snapshot->fingerprint() == snapshot_fingerprint);
  std::vector<std::uint8_t> corrupted_snapshot = encoded_snapshot;
  corrupted_snapshot.back() ^= 0x01;
  bool corruption_rejected = false;
  try
  {
    VoxelMapSnapshot::deserialize(corrupted_snapshot);
  }
  catch (const std::runtime_error &)
  {
    corruption_rejected = true;
  }
  assert(corruption_rejected);
  bool truncated_rejected = false;
  try
  {
    corrupted_snapshot.resize(16);
    VoxelMapSnapshot::deserialize(corrupted_snapshot);
  }
  catch (const std::runtime_error &)
  {
    truncated_rejected = true;
  }
  assert(truncated_rejected);

  // Simulate both in-place corruption and insertion of a polluted voxel.
  verified_root->plane_ptr_->center_ = Eigen::Vector3d(100.0, 200.0, 300.0);
  manager.voxel_map_.emplace(VOXEL_LOCATION(9, 9, 9),
                             new VoxelOctoTree(2, 0, 5, 50, 0.01f));
  assert(manager.voxel_map_.size() == 2);

  manager.RestoreSnapshot(*snapshot);
  assert(manager.voxel_map_.size() == 1);
  const auto restored = manager.voxel_map_.find(verified_location);
  assert(restored != manager.voxel_map_.end());
  // Do not compare raw addresses: the allocator may legally reuse the freed
  // live-root address for the deep clone. Restored values plus the second
  // mutation/restore cycle below prove snapshot isolation.
  assert(restored->second->plane_ptr_->center_.isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
  assert(restored->second->leaves_[0] != nullptr);
  assert(restored->second->leaves_[0]->plane_ptr_->points_size_ == 7);

  // A memory-bounded restore releases the live map before cloning, but the
  // verified snapshot must remain reusable for candidate-rejection rebuilds.
  restored->second->plane_ptr_->center_ = Eigen::Vector3d(-4.0, -5.0, -6.0);
  manager.voxel_map_.emplace(VOXEL_LOCATION(7, 8, 9),
                             new VoxelOctoTree(2, 0, 5, 50, 0.01f));
  manager.RestoreSnapshot(*snapshot);
  assert(manager.voxel_map_.size() == 1);
  const auto restored_again = manager.voxel_map_.find(verified_location);
  assert(restored_again != manager.voxel_map_.end());
  assert(restored_again->second->plane_ptr_->center_.isApprox(
      Eigen::Vector3d(1.0, 2.0, 3.0)));

  // Process-isolated shadow recovery can transfer the verified generation
  // directly, avoiding the second deep copy used by reusable restoration.
  std::unique_ptr<VoxelMapSnapshot> movable_snapshot = manager.CreateSnapshot();
  manager.voxel_map_.emplace(VOXEL_LOCATION(11, 12, 13),
                             new VoxelOctoTree(2, 0, 5, 50, 0.01f));
  manager.ConsumeSnapshot(*movable_snapshot);
  assert(movable_snapshot->size() == 0);
  assert(manager.voxel_map_.size() == 1);
  assert(manager.voxel_map_.find(verified_location) != manager.voxel_map_.end());

  // Common-reference evaluation must score two poses against the exact same
  // immutable geometry, independent of later mutations to the live map.
  manager.ClearVoxelMap();
  VoxelOctoTree *reference_root =
      new VoxelOctoTree(2, 0, 5, 50, 0.01f);
  reference_root->voxel_center_[0] = 0.5;
  reference_root->voxel_center_[1] = 0.5;
  reference_root->voxel_center_[2] = 0.5;
  reference_root->quater_length_ = 0.25;
  reference_root->plane_ptr_->center_ = Eigen::Vector3d(0.5, 0.5, 0.0);
  reference_root->plane_ptr_->normal_ = Eigen::Vector3d(0.0, 0.0, 1.0);
  reference_root->plane_ptr_->x_normal_ = Eigen::Vector3d(1.0, 0.0, 0.0);
  reference_root->plane_ptr_->y_normal_ = Eigen::Vector3d(0.0, 1.0, 0.0);
  reference_root->plane_ptr_->max_eigen_value_ = 0.04f;
  reference_root->plane_ptr_->mid_eigen_value_ = 0.01f;
  reference_root->plane_ptr_->points_size_ = 20;
  reference_root->plane_ptr_->d_ = 0.0;
  reference_root->plane_ptr_->radius_ = 10.0;
  reference_root->plane_ptr_->plane_var_ =
      Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
  reference_root->plane_ptr_->is_plane_ = true;
  manager.voxel_map_.emplace(VOXEL_LOCATION(0, 0, 0), reference_root);
  manager.extR_ = Eigen::Matrix3d::Identity();
  manager.extT_ = Eigen::Vector3d::Zero();
  std::unique_ptr<VoxelMapSnapshot> reference_snapshot =
      manager.CreateSnapshot();

  PointCloudXYZI::Ptr body_points(new PointCloudXYZI());
  PointType point;
  // Keep a non-zero tangential distance so the production float calculation
  // of sqrt(distance-to-center^2 - distance-to-plane^2) stays well away from
  // a tiny negative round-off at exactly the plane center.
  point.x = 0.4f;
  point.y = 0.4f;
  point.z = 0.05f;
  body_points->push_back(point);
  StatesGroup near_state;
  StatesGroup shifted_state;
  shifted_state.pos_end.z() = 0.01;
  const auto near_statistics = manager.EvaluateResidualsAgainstSnapshot(
      *reference_snapshot, near_state, body_points);
  const auto shifted_statistics = manager.EvaluateResidualsAgainstSnapshot(
      *reference_snapshot, shifted_state, body_points);
  assert(near_statistics.valid);
  assert(shifted_statistics.valid);
  assert(near_statistics.matched_points == 1);
  assert(shifted_statistics.mean_abs_residual >
         near_statistics.mean_abs_residual);

  reference_root->plane_ptr_->d_ = 100.0;
  const auto immutable_statistics = manager.EvaluateResidualsAgainstSnapshot(
      *reference_snapshot, near_state, body_points);
  assert(immutable_statistics.valid);
  assert(std::abs(immutable_statistics.mean_abs_residual -
                  near_statistics.mean_abs_residual) < 1e-9);

  // Pose refinement uses the immutable snapshot and must improve a supported
  // plane residual without touching either the snapshot or the live map.
  PointCloudXYZI::Ptr supported_points(new PointCloudXYZI());
  for (int index = 0; index < 40; ++index)
  {
    PointType sample = point;
    sample.x += 0.001f * (index % 5);
    sample.y += 0.001f * (index / 5);
    supported_points->push_back(sample);
  }
  const std::uint64_t reference_fingerprint = reference_snapshot->fingerprint();
  const std::size_t live_root_count = manager.voxel_map_.size();
  const auto refinement = manager.RefinePoseAgainstSnapshot(
      *reference_snapshot, near_state, supported_points);
  assert(refinement.initial.matched_points >= 30);
  assert(refinement.accepted_steps > 0);
  assert(refinement.final.mean_abs_residual <
         refinement.initial.mean_abs_residual);
  assert((refinement.state.pos_end - near_state.pos_end).norm() <= 0.20);
  assert(reference_snapshot->fingerprint() == reference_fingerprint);
  assert(manager.voxel_map_.size() == live_root_count);

  PointCloudXYZI::Ptr corrected_points(new PointCloudXYZI(*supported_points));
  for (PointType &sample : corrected_points->points) sample.z -= 0.02f;
  const auto common_planes = manager.EvaluateCommonPlanesAgainstSnapshot(
      *reference_snapshot, near_state, supported_points,
      near_state, corrected_points);
  assert(common_planes.valid);
  assert(common_planes.shared_points == supported_points->size());
  assert(common_planes.reference_matches >= 30);
  assert(common_planes.candidate_mean_abs_residual <
         common_planes.reference_mean_abs_residual);
  assert(reference_snapshot->fingerprint() == reference_fingerprint);
  corrected_points->points.pop_back();
  assert(!manager.EvaluateCommonPlanesAgainstSnapshot(
      *reference_snapshot, near_state, supported_points,
      near_state, corrected_points).valid);

  // The bounded exporter reads the final voxel map directly, so its memory is
  // independent of scan duration. Root ordering and per-plane sample ordering
  // are deterministic, and the explicit cap is enforced without overshoot.
  VoxelOctoTree *fallback_root =
      new VoxelOctoTree(2, 0, 5, 50, 0.01f);
  pointWithVar fallback_point;
  fallback_point.point_w = Eigen::Vector3d(2.5, 0.5, 0.25);
  fallback_root->temp_points_.push_back(fallback_point);
  manager.voxel_map_.emplace(VOXEL_LOCATION(2, 0, 0), fallback_root);

  PointCloudXYZI first_export;
  const auto complete_export = manager.ExportRepresentativeCloud(
      first_export, 20, 9);
  assert(!complete_export.truncated);
  assert(complete_export.root_voxels == 2);
  assert(complete_export.plane_nodes == 1);
  assert(complete_export.fallback_nodes == 1);
  assert(complete_export.emitted_points == 10);

  PointCloudXYZI repeated_export;
  const auto repeated_statistics = manager.ExportRepresentativeCloud(
      repeated_export, 20, 9);
  assert(repeated_statistics.emitted_points == complete_export.emitted_points);
  assert(repeated_export.size() == first_export.size());
  for (std::size_t index = 0; index < first_export.size(); ++index)
  {
    assert(first_export[index].x == repeated_export[index].x);
    assert(first_export[index].y == repeated_export[index].y);
    assert(first_export[index].z == repeated_export[index].z);
  }

  PointCloudXYZI capped_export;
  const auto capped_statistics = manager.ExportRepresentativeCloud(
      capped_export, 5, 9);
  assert(!capped_statistics.truncated);
  assert(capped_export.size() == 5);
  assert(capped_statistics.emitted_points == 5);

  manager.ClearVoxelMap();
  std::cout << "VOXEL_MAP_SNAPSHOT_TEST_OK" << std::endl;
  return 0;
}
