#include "IMU_Processing.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main()
{
  ImuProcess::Checkpoint checkpoint;
  checkpoint.pcl_wait_proc.header.seq = 7;
  checkpoint.pcl_wait_proc.header.stamp = 123456789ULL;
  checkpoint.pcl_wait_proc.header.frame_id = "lidar";
  PointType point{};
  point.x = 1.25f;
  point.y = -2.5f;
  point.z = 3.75f;
  point.intensity = 42.0f;
  point.normal_x = 0.1f;
  point.normal_y = 0.2f;
  point.normal_z = 0.3f;
  point.curvature = 4.0f;
  checkpoint.pcl_wait_proc.push_back(point);
  checkpoint.pcl_wait_proc.width = 1;
  checkpoint.pcl_wait_proc.height = 1;
  checkpoint.pcl_wait_proc.is_dense = true;
  checkpoint.has_last_imu = true;
  checkpoint.last_imu.header.seq = 9;
  checkpoint.last_imu.header.stamp = ros::Time(12, 345);
  checkpoint.last_imu.header.frame_id = "imu";
  checkpoint.last_imu.orientation.w = 1.0;
  checkpoint.last_imu.angular_velocity.z = 0.125;
  checkpoint.last_imu.linear_acceleration.z = -9.81;
  checkpoint.last_imu.linear_acceleration_covariance[8] = 0.04;
  Pose6D pose{};
  pose.offset_time = 0.01;
  pose.acc[2] = -9.8;
  pose.rot[0] = pose.rot[4] = pose.rot[8] = 1.0;
  checkpoint.imu_poses.push_back(pose);
  checkpoint.mean_acc = V3D(1.0, 2.0, 3.0);
  checkpoint.mean_gyr = V3D(4.0, 5.0, 6.0);
  checkpoint.angvel_last = V3D(7.0, 8.0, 9.0);
  checkpoint.acc_s_last = V3D(10.0, 11.0, 12.0);
  checkpoint.last_prop_end_time = 15.5;
  checkpoint.time_last_scan = 14.5;
  checkpoint.imu_mean_acc_norm = 9.805;
  checkpoint.unbiased_gyr = V3D(0.01, 0.02, 0.03);
  checkpoint.init_iter_num = 21;
  checkpoint.first_frame = false;
  checkpoint.imu_time_initialized = true;
  checkpoint.imu_needs_initialization = false;

  const std::vector<std::uint8_t> encoded =
      ImuProcess::serializeCheckpoint(checkpoint);
  const ImuProcess::Checkpoint decoded =
      ImuProcess::deserializeCheckpoint(encoded);
  assert(ImuProcess::serializeCheckpoint(decoded) == encoded);
  assert(ImuProcess::checkpointFingerprint(decoded) ==
         ImuProcess::checkpointFingerprint(checkpoint));
  assert(decoded.pcl_wait_proc.front().curvature == 4.0f);
  assert(decoded.last_imu.header.frame_id == "imu");
  assert(decoded.last_imu.linear_acceleration.z == -9.81);
  assert(decoded.imu_poses.front().rot[8] == 1.0);
  assert(decoded.mean_acc == V3D(1.0, 2.0, 3.0));
  assert(decoded.imu_time_initialized);

  std::vector<std::uint8_t> corrupted = encoded;
  corrupted[20] ^= 0x40;
  bool rejected = false;
  try
  {
    ImuProcess::deserializeCheckpoint(corrupted);
  }
  catch (const std::runtime_error &)
  {
    rejected = true;
  }
  assert(rejected);
  std::cout << "IMU_CHECKPOINT_SERIALIZATION_TEST_OK" << std::endl;
  return 0;
}
