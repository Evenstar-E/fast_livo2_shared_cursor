#include "LIVMapper.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main()
{
  LIVMapper::MapperCheckpoint checkpoint;
  checkpoint.state.pos_end = V3D(1.0, 2.0, 3.0);
  checkpoint.state.bias_g = V3D(0.01, 0.02, 0.03);
  checkpoint.state.cov(3, 3) = 0.125;
  checkpoint.propagated_state.vel_end = V3D(4.0, 5.0, 6.0);
  checkpoint.latest_ekf_state.inv_expo_time = 1.25;
  checkpoint.imu_propagate.gravity = V3D(0.0, 0.0, -9.81);
  checkpoint.imu_process.mean_acc = V3D(0.1, 0.2, 0.3);
  checkpoint.imu_process.first_frame = false;
  pointWithVar variance;
  variance.point_b = V3D(7.0, 8.0, 9.0);
  variance.var = M3D::Identity() * 0.5;
  checkpoint.point_variances.push_back(variance);
  PointType point{};
  point.x = 1.0f;
  point.y = -2.0f;
  point.z = 3.0f;
  point.intensity = 4.0f;
  checkpoint.lidar_world_for_visual.push_back(point);
  checkpoint.lidar_world_for_visual.width = 1;
  checkpoint.lidar_world_for_visual.height = 1;
  checkpoint.pending_publish_cloud.push_back(point);
  checkpoint.pending_publish_cloud.width = 1;
  checkpoint.pending_publish_cloud.height = 1;
  checkpoint.voxel_position_last = V3D(10.0, 11.0, 12.0);
  checkpoint.voxel_last_slide_position = V3D(13.0, 14.0, 15.0);
  checkpoint.latest_ekf_time = 101.5;
  checkpoint.last_completed_event_relative_time = 99.5;
  checkpoint.average_processing_time = 0.021;
  checkpoint.path_pose_count = 101;
  checkpoint.trajectory_pose_count = 102;
  checkpoint.pcd_rgb_point_count = 103;
  checkpoint.pcd_intensity_point_count = 104;
  checkpoint.voxel_frame_id = 105;
  checkpoint.visual_frame_counter = 106;
  checkpoint.visual_processed_frames = 107;
  checkpoint.processed_lidar_frames = 108;
  checkpoint.lidar_map_inited = true;
  checkpoint.ekf_finish_once = true;
  checkpoint.state_update_flg = false;

  const std::vector<std::uint8_t> encoded =
      LIVMapper::serializeMapperCheckpoint(checkpoint);
  const LIVMapper::MapperCheckpoint decoded =
      LIVMapper::deserializeMapperCheckpoint(encoded);
  assert(LIVMapper::serializeMapperCheckpoint(decoded) == encoded);
  assert(LIVMapper::mapperCheckpointFingerprint(decoded) ==
         LIVMapper::mapperCheckpointFingerprint(checkpoint));
  assert(decoded.state.pos_end == V3D(1.0, 2.0, 3.0));
  assert(decoded.state.cov(3, 3) == 0.125);
  assert(decoded.point_variances.front().var(0, 0) == 0.5);
  assert(decoded.lidar_world_for_visual.front().intensity == 4.0f);
  assert(decoded.last_completed_event_relative_time == 99.5);
  assert(decoded.visual_frame_counter == 106);
  assert(decoded.lidar_map_inited);

  std::vector<std::uint8_t> corrupted = encoded;
  corrupted[encoded.size() / 3] ^= 0x01;
  bool rejected = false;
  try
  {
    LIVMapper::deserializeMapperCheckpoint(corrupted);
  }
  catch (const std::runtime_error &)
  {
    rejected = true;
  }
  assert(rejected);
  std::cout << "MAPPER_CHECKPOINT_SERIALIZATION_TEST_OK" << std::endl;
  return 0;
}
