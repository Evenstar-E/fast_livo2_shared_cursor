/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "IMU_Processing.h"
#include "checkpoint_binary_codec.h"

#include <cstring>
#include <stdexcept>

namespace
{
using fast_livo::checkpoint_io::Reader;
using fast_livo::checkpoint_io::Writer;

constexpr std::uint64_t kImuCheckpointMagic = 0x31504b43554d4946ULL; // FIMUCKP1
constexpr std::uint32_t kImuCheckpointSchemaVersion = 1;

void writePoint(Writer &writer, const PointType &point)
{
  writer.pod(point.x);
  writer.pod(point.y);
  writer.pod(point.z);
  writer.pod(point.intensity);
  writer.pod(point.normal_x);
  writer.pod(point.normal_y);
  writer.pod(point.normal_z);
  writer.pod(point.curvature);
}

PointType readPoint(Reader &reader)
{
  PointType point{};
  point.x = reader.pod<float>();
  point.y = reader.pod<float>();
  point.z = reader.pod<float>();
  point.intensity = reader.pod<float>();
  point.normal_x = reader.pod<float>();
  point.normal_y = reader.pod<float>();
  point.normal_z = reader.pod<float>();
  point.curvature = reader.pod<float>();
  return point;
}

void writeCloud(Writer &writer, const PointCloudXYZI &cloud)
{
  writer.pod(cloud.header.seq);
  writer.pod(cloud.header.stamp);
  writer.string(cloud.header.frame_id);
  writer.pod(cloud.width);
  writer.pod(cloud.height);
  writer.pod(static_cast<std::uint8_t>(cloud.is_dense));
  writer.eigen(cloud.sensor_origin_);
  writer.eigen(cloud.sensor_orientation_.coeffs());
  writer.pod(static_cast<std::uint64_t>(cloud.points.size()));
  for (const PointType &point : cloud.points) writePoint(writer, point);
}

PointCloudXYZI readCloud(Reader &reader)
{
  PointCloudXYZI cloud;
  cloud.header.seq = reader.pod<std::uint32_t>();
  cloud.header.stamp = reader.pod<std::uint64_t>();
  cloud.header.frame_id = reader.string();
  cloud.width = reader.pod<std::uint32_t>();
  cloud.height = reader.pod<std::uint32_t>();
  cloud.is_dense = reader.pod<std::uint8_t>() != 0;
  reader.eigen(cloud.sensor_origin_);
  Eigen::Vector4f quaternion_coefficients;
  reader.eigen(quaternion_coefficients);
  cloud.sensor_orientation_.coeffs() = quaternion_coefficients;
  const std::uint64_t count = fast_livo::checkpoint_io::checkedCount(
      reader, 1000000000, "IMU pending-cloud point");
  cloud.points.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index)
    cloud.points.push_back(readPoint(reader));
  if (cloud.points.size() != static_cast<std::size_t>(cloud.width) * cloud.height)
    throw std::runtime_error("IMU checkpoint cloud shape mismatch");
  return cloud;
}

void writeImuMessage(Writer &writer, const sensor_msgs::Imu &message)
{
  writer.pod(message.header.seq);
  writer.pod(message.header.stamp.sec);
  writer.pod(message.header.stamp.nsec);
  writer.string(message.header.frame_id);
  writer.pod(message.orientation.x);
  writer.pod(message.orientation.y);
  writer.pod(message.orientation.z);
  writer.pod(message.orientation.w);
  for (const double value : message.orientation_covariance) writer.pod(value);
  writer.pod(message.angular_velocity.x);
  writer.pod(message.angular_velocity.y);
  writer.pod(message.angular_velocity.z);
  for (const double value : message.angular_velocity_covariance) writer.pod(value);
  writer.pod(message.linear_acceleration.x);
  writer.pod(message.linear_acceleration.y);
  writer.pod(message.linear_acceleration.z);
  for (const double value : message.linear_acceleration_covariance) writer.pod(value);
}

sensor_msgs::Imu readImuMessage(Reader &reader)
{
  sensor_msgs::Imu message;
  message.header.seq = reader.pod<std::uint32_t>();
  message.header.stamp.sec = reader.pod<std::uint32_t>();
  message.header.stamp.nsec = reader.pod<std::uint32_t>();
  message.header.frame_id = reader.string();
  message.orientation.x = reader.pod<double>();
  message.orientation.y = reader.pod<double>();
  message.orientation.z = reader.pod<double>();
  message.orientation.w = reader.pod<double>();
  for (double &value : message.orientation_covariance) value = reader.pod<double>();
  message.angular_velocity.x = reader.pod<double>();
  message.angular_velocity.y = reader.pod<double>();
  message.angular_velocity.z = reader.pod<double>();
  for (double &value : message.angular_velocity_covariance) value = reader.pod<double>();
  message.linear_acceleration.x = reader.pod<double>();
  message.linear_acceleration.y = reader.pod<double>();
  message.linear_acceleration.z = reader.pod<double>();
  for (double &value : message.linear_acceleration_covariance) value = reader.pod<double>();
  return message;
}
} // namespace

ImuProcess::ImuProcess() : Eye3d(M3D::Identity()),
                           Zero3d(0, 0, 0), b_first_frame(true), imu_need_init(true)
{
  init_iter_num = 1;
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_acc = V3D(0.1, 0.1, 0.1);
  cov_inv_expo = 0.2;
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  acc_s_last = Zero3d;
  Lid_offset_to_IMU = Zero3d;
  Lid_rot_to_IMU = Eye3d;
  last_imu.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
  last_prop_end_time = 0.0;
  time_last_scan = 0.0;
  IMU_mean_acc_norm = 0.0;
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()
{
  ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init = true;
  init_iter_num = 1;
  IMUpose.clear();
  last_imu.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::disable_imu()
{
  cout << "IMU Disabled !!!!!" << endl;
  imu_en = false;
  imu_need_init = false;
}

void ImuProcess::disable_gravity_est()
{
  cout << "Online Gravity Estimation Disabled !!!!!" << endl;
  gravity_est_en = false;
}

void ImuProcess::disable_bias_est()
{
  cout << "Bias Estimation Disabled !!!!!" << endl;
  ba_bg_est_en = false;
}

void ImuProcess::disable_exposure_est()
{
  cout << "Online Time Offset Estimation Disabled !!!!!" << endl;
  exposure_estimate_en = false;
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T)
{
  Lid_offset_to_IMU = T.block<3, 1>(0, 3);
  Lid_rot_to_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU = rot;
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler) { cov_gyr = scaler; }

void ImuProcess::set_acc_cov_scale(const V3D &scaler) { cov_acc = scaler; }

void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }

void ImuProcess::set_inv_expo_cov(const double &inv_expo) { cov_inv_expo = inv_expo; }

void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }

void ImuProcess::set_imu_init_frame_num(const int &num) { MAX_INI_COUNT = num; }

ImuProcess::Checkpoint ImuProcess::createCheckpoint() const
{
  Checkpoint checkpoint;
  checkpoint.pcl_wait_proc = pcl_wait_proc;
  checkpoint.has_last_imu = static_cast<bool>(last_imu);
  if (checkpoint.has_last_imu) checkpoint.last_imu = *last_imu;
  checkpoint.imu_poses = IMUpose;
  checkpoint.mean_acc = mean_acc;
  checkpoint.mean_gyr = mean_gyr;
  checkpoint.angvel_last = angvel_last;
  checkpoint.acc_s_last = acc_s_last;
  checkpoint.last_prop_end_time = last_prop_end_time;
  checkpoint.time_last_scan = time_last_scan;
  checkpoint.imu_mean_acc_norm = IMU_mean_acc_norm;
  checkpoint.unbiased_gyr = unbiased_gyr;
  checkpoint.init_iter_num = init_iter_num;
  checkpoint.first_frame = b_first_frame;
  checkpoint.imu_time_initialized = imu_time_init;
  checkpoint.imu_needs_initialization = imu_need_init;
  return checkpoint;
}

std::vector<std::uint8_t> ImuProcess::serializeCheckpoint(
    const Checkpoint &checkpoint)
{
  Writer writer;
  writer.pod(kImuCheckpointMagic);
  writer.pod(kImuCheckpointSchemaVersion);
  writeCloud(writer, checkpoint.pcl_wait_proc);
  writer.pod(static_cast<std::uint8_t>(checkpoint.has_last_imu));
  if (checkpoint.has_last_imu) writeImuMessage(writer, checkpoint.last_imu);
  writer.pod(static_cast<std::uint64_t>(checkpoint.imu_poses.size()));
  for (const Pose6D &pose : checkpoint.imu_poses)
  {
    writer.pod(pose.offset_time);
    for (const double value : pose.acc) writer.pod(value);
    for (const double value : pose.gyr) writer.pod(value);
    for (const double value : pose.vel) writer.pod(value);
    for (const double value : pose.pos) writer.pod(value);
    for (const double value : pose.rot) writer.pod(value);
  }
  writer.eigen(checkpoint.mean_acc);
  writer.eigen(checkpoint.mean_gyr);
  writer.eigen(checkpoint.angvel_last);
  writer.eigen(checkpoint.acc_s_last);
  writer.pod(checkpoint.last_prop_end_time);
  writer.pod(checkpoint.time_last_scan);
  writer.pod(checkpoint.imu_mean_acc_norm);
  writer.eigen(checkpoint.unbiased_gyr);
  writer.pod(checkpoint.init_iter_num);
  writer.pod(static_cast<std::uint8_t>(checkpoint.first_frame));
  writer.pod(static_cast<std::uint8_t>(checkpoint.imu_time_initialized));
  writer.pod(static_cast<std::uint8_t>(checkpoint.imu_needs_initialization));
  const std::uint64_t checksum =
      fast_livo::checkpoint_io::fnv1a(writer.bytes());
  writer.pod(checksum);
  return writer.take();
}

ImuProcess::Checkpoint ImuProcess::deserializeCheckpoint(
    const std::vector<std::uint8_t> &payload)
{
  if (payload.size() < sizeof(std::uint64_t))
    throw std::runtime_error("truncated IMU checkpoint checksum");
  std::uint64_t stored_checksum = 0;
  std::memcpy(&stored_checksum,
              payload.data() + payload.size() - sizeof(stored_checksum),
              sizeof(stored_checksum));
  const std::vector<std::uint8_t> body(
      payload.begin(), payload.end() - sizeof(stored_checksum));
  if (fast_livo::checkpoint_io::fnv1a(body) != stored_checksum)
    throw std::runtime_error("IMU checkpoint checksum mismatch");
  Reader reader(body);
  if (reader.pod<std::uint64_t>() != kImuCheckpointMagic)
    throw std::runtime_error("invalid IMU checkpoint magic");
  if (reader.pod<std::uint32_t>() != kImuCheckpointSchemaVersion)
    throw std::runtime_error("unsupported IMU checkpoint schema");
  Checkpoint checkpoint;
  checkpoint.pcl_wait_proc = readCloud(reader);
  checkpoint.has_last_imu = reader.pod<std::uint8_t>() != 0;
  if (checkpoint.has_last_imu) checkpoint.last_imu = readImuMessage(reader);
  const std::uint64_t pose_count = fast_livo::checkpoint_io::checkedCount(
      reader, 100000000, "IMU pose");
  checkpoint.imu_poses.resize(static_cast<std::size_t>(pose_count));
  for (Pose6D &pose : checkpoint.imu_poses)
  {
    pose.offset_time = reader.pod<double>();
    for (double &value : pose.acc) value = reader.pod<double>();
    for (double &value : pose.gyr) value = reader.pod<double>();
    for (double &value : pose.vel) value = reader.pod<double>();
    for (double &value : pose.pos) value = reader.pod<double>();
    for (double &value : pose.rot) value = reader.pod<double>();
  }
  reader.eigen(checkpoint.mean_acc);
  reader.eigen(checkpoint.mean_gyr);
  reader.eigen(checkpoint.angvel_last);
  reader.eigen(checkpoint.acc_s_last);
  checkpoint.last_prop_end_time = reader.pod<double>();
  checkpoint.time_last_scan = reader.pod<double>();
  checkpoint.imu_mean_acc_norm = reader.pod<double>();
  reader.eigen(checkpoint.unbiased_gyr);
  checkpoint.init_iter_num = reader.pod<int>();
  checkpoint.first_frame = reader.pod<std::uint8_t>() != 0;
  checkpoint.imu_time_initialized = reader.pod<std::uint8_t>() != 0;
  checkpoint.imu_needs_initialization = reader.pod<std::uint8_t>() != 0;
  if (!reader.finished())
    throw std::runtime_error("trailing bytes in IMU checkpoint");
  return checkpoint;
}

std::uint64_t ImuProcess::checkpointFingerprint(const Checkpoint &checkpoint)
{
  return fast_livo::checkpoint_io::fnv1a(serializeCheckpoint(checkpoint));
}

void ImuProcess::restoreCheckpoint(const Checkpoint &checkpoint)
{
  pcl_wait_proc = checkpoint.pcl_wait_proc;
  if (checkpoint.has_last_imu)
  {
    last_imu.reset(new sensor_msgs::Imu(checkpoint.last_imu));
  }
  else
  {
    last_imu.reset();
  }
  IMUpose = checkpoint.imu_poses;
  mean_acc = checkpoint.mean_acc;
  mean_gyr = checkpoint.mean_gyr;
  angvel_last = checkpoint.angvel_last;
  acc_s_last = checkpoint.acc_s_last;
  last_prop_end_time = checkpoint.last_prop_end_time;
  time_last_scan = checkpoint.time_last_scan;
  IMU_mean_acc_norm = checkpoint.imu_mean_acc_norm;
  unbiased_gyr = checkpoint.unbiased_gyr;
  init_iter_num = checkpoint.init_iter_num;
  b_first_frame = checkpoint.first_frame;
  imu_time_init = checkpoint.imu_time_initialized;
  imu_need_init = checkpoint.imu_needs_initialization;
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;

  if (b_first_frame)
  {
    Reset();
    N = 1;
    b_first_frame = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    // first_lidar_time = meas.lidar_frame_beg_time;
    // cout<<"init acc norm: "<<mean_acc.norm()<<endl;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    // cov_acc = cov_acc * (N - 1.0) / N + (cur_acc -
    // mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N); cov_gyr
    // = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr -
    // mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N++;
  }
  IMU_mean_acc_norm = mean_acc.norm();
  state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;
  state_inout.rot_end = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  state_inout.bias_g = Zero3d; // mean_gyr;

  last_imu = meas.imu.back();
}

void ImuProcess::Forward_without_imu(LidarMeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  pcl_out = *(meas.lidar);
  /*** sort point clouds by offset time ***/
  const double &pcl_beg_time = meas.lidar_frame_beg_time;
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  meas.last_lio_update_time = pcl_end_time;
  const double &pcl_end_offset_time = pcl_out.points.back().curvature / double(1000);

  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt = 0;

  if (b_first_frame)
  {
    dt = 0.1;
    b_first_frame = false;
  }
  else { dt = pcl_beg_time - time_last_scan; }

  time_last_scan = pcl_beg_time;
  // for (size_t i = 0; i < pcl_out->points.size(); i++) {
  //   if (dt < pcl_out->points[i].curvature) {
  //     dt = pcl_out->points[i].curvature;
  //   }
  // }
  // dt = dt / (double)1000;
  // std::cout << "dt:" << dt << std::endl;
  // double dt = pcl_out->points.back().curvature / double(1000);

  /* covariance propagation */
  // M3D acc_avr_skew;
  M3D Exp_f = Exp(state_inout.bias_g, dt);

  F_x.setIdentity();
  cov_w.setZero();

  F_x.block<3, 3>(0, 0) = Exp(state_inout.bias_g, -dt);
  F_x.block<3, 3>(0, 10) = Eye3d * dt;
  F_x.block<3, 3>(3, 7) = Eye3d * dt;
  // F_x.block<3, 3>(6, 0)  = - R_imu * acc_avr_skew * dt;
  // F_x.block<3, 3>(6, 12) = - R_imu * dt;
  // F_x.block<3, 3>(6, 15) = Eye3d * dt;

  cov_w.block<3, 3>(10, 10).diagonal() = cov_gyr * dt * dt; // for omega in constant model
  cov_w.block<3, 3>(7, 7).diagonal() = cov_acc * dt * dt; // for velocity in constant model
  // cov_w.block<3, 3>(6, 6) =
  //     R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
  // cov_w.block<3, 3>(9, 9).diagonal() =
  //     cov_bias_gyr * dt * dt; // bias gyro covariance
  // cov_w.block<3, 3>(12, 12).diagonal() =
  //     cov_bias_acc * dt * dt; // bias acc covariance

  // std::cout << "before propagete:" << state_inout.cov.diagonal().transpose()
  //           << std::endl;
  state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
  // std::cout << "cov_w:" << cov_w.diagonal().transpose() << std::endl;
  // std::cout << "after propagete:" << state_inout.cov.diagonal().transpose()
  //           << std::endl;
  state_inout.rot_end = state_inout.rot_end * Exp_f;
  state_inout.pos_end = state_inout.pos_end + state_inout.vel_end * dt;

  if (lidar_type != L515)
  {
    auto it_pcl = pcl_out.points.end() - 1;
    double dt_j = 0.0;
    for(; it_pcl != pcl_out.points.begin(); it_pcl--)
    {
        dt_j= pcl_end_offset_time - it_pcl->curvature/double(1000);
        M3D R_jk(Exp(state_inout.bias_g, - dt_j));
        V3D P_j(it_pcl->x, it_pcl->y, it_pcl->z);
        // Using rotation and translation to un-distort points
        V3D p_jk;
        p_jk = - state_inout.rot_end.transpose() * state_inout.vel_end * dt_j;
  
        V3D P_compensate =  R_jk * P_j + p_jk;
  
        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);
    }
  }
}


void ImuProcess::UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
  double t0 = omp_get_wtime();
  pcl_out.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time;
  // printf("[ IMU ] undistort input size: %zu \n", lidar_meas.pcl_proc_cur->points.size());
  // printf("[ IMU ] IMU data sequence size: %zu \n", meas.imu.size());
  // printf("[ IMU ] lidar_scan_index_now: %d \n", lidar_meas.lidar_scan_index_now);

  const double prop_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

  /*** cut lidar point based on the propagation-start time and required
   * propagation-end time ***/
  // const double pcl_offset_time = (prop_end_time -
  // lidar_meas.lidar_frame_beg_time) * 1000.; // the offset time w.r.t scan
  // start time auto pcl_it = lidar_meas.pcl_proc_cur->points.begin() +
  // lidar_meas.lidar_scan_index_now; auto pcl_it_end =
  // lidar_meas.lidar->points.end(); printf("[ IMU ] pcl_it->curvature: %lf
  // pcl_offset_time: %lf \n", pcl_it->curvature, pcl_offset_time); while
  // (pcl_it != pcl_it_end && pcl_it->curvature <= pcl_offset_time)
  // {
  //   pcl_wait_proc.push_back(*pcl_it);
  //   pcl_it++;
  //   lidar_meas.lidar_scan_index_now++;
  // }

  // cout<<"pcl_out.size(): "<<pcl_out.size()<<endl;
  // cout<<"pcl_offset_time:  "<<pcl_offset_time<<"pcl_it->curvature:
  // "<<pcl_it->curvature<<endl;
  // cout<<"lidar_meas.lidar_scan_index_now:"<<lidar_meas.lidar_scan_index_now<<endl;

  // printf("[ IMU ] last propagation end time: %lf \n", lidar_meas.last_lio_update_time);
  if (lidar_meas.lio_vio_flg == LIO)
  {
    pcl_wait_proc.resize(lidar_meas.pcl_proc_cur->points.size());
    pcl_wait_proc = *(lidar_meas.pcl_proc_cur);
    lidar_meas.lidar_scan_index_now = 0;
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));
  }

  // printf("[ IMU ] pcl_wait_proc size: %zu \n", pcl_wait_proc.points.size());

  // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // lidar_meas.debug_show();
  // cout<<"UndistortPcl [ IMU ]: Process lidar from "<<prop_beg_time<<" to
  // "<<prop_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to
  //          "<<imu_end_time<<endl;
  // cout<<"[ IMU ]: point size: "<<lidar_meas.lidar->points.size()<<endl;

  /*** Initialize IMU pose ***/
  // IMUpose.clear();

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  // cout << "[ IMU ] input state: " << state_inout.vel_end.transpose() << " " << state_inout.pos_end.transpose() << endl;
  M3D R_imu(state_inout.rot_end);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt, dt_all = 0.0;
  double offs_t;
  // double imu_time;
  double tau;
  if (!imu_time_init)
  {
    // imu_time = v_imu.front()->header.stamp.toSec() - first_lidar_time;
    // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
    tau = 1.0;
    imu_time_init = true;
  }
  else
  {
    tau = state_inout.inv_expo_time;
    // ROS_ERROR("tau: %.6f !!!!!!", tau);
  }
  // state_inout.cov(6, 6) = 0.01;

  // ROS_ERROR("lidar_meas.lio_vio_flg");
  // cout<<"lidar_meas.lio_vio_flg: "<<lidar_meas.lio_vio_flg<<endl;
  switch (lidar_meas.lio_vio_flg)
  {
  case LIO:
  case VIO:
    dt = 0;
    for (int i = 0; i < v_imu.size() - 1; i++)
    {
      auto head = v_imu[i];
      auto tail = v_imu[i + 1];

      if (tail->header.stamp.toSec() < prop_beg_time) continue;

      angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), 0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
          0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

      // angvel_avr<<tail->angular_velocity.x, tail->angular_velocity.y,
      // tail->angular_velocity.z;

      acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
          0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

      // cout<<"angvel_avr: "<<angvel_avr.transpose()<<endl;
      // cout<<"acc_avr: "<<acc_avr.transpose()<<endl;

      // #ifdef DEBUG_PRINT
      fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
      // #endif

      // imu_time = head->header.stamp.toSec() - first_lidar_time;

      angvel_avr -= state_inout.bias_g;
      acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

      if (head->header.stamp.toSec() < prop_beg_time)
      {
        // printf("00 \n");
        dt = tail->header.stamp.toSec() - last_prop_end_time;
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else if (i != v_imu.size() - 2)
      {
        // printf("11 \n");
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        offs_t = tail->header.stamp.toSec() - prop_beg_time;
      }
      else
      {
        // printf("22 \n");
        dt = prop_end_time - head->header.stamp.toSec();
        offs_t = prop_end_time - prop_beg_time;
      }

      dt_all += dt;
      // printf("[ LIO Propagation ] dt: %lf \n", dt);

      /* covariance propagation */
      M3D acc_avr_skew;
      M3D Exp_f = Exp(angvel_avr, dt);
      acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

      F_x.setIdentity();
      cov_w.setZero();

      F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
      if (ba_bg_est_en) F_x.block<3, 3>(0, 10) = -Eye3d * dt;
      // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
      F_x.block<3, 3>(3, 7) = Eye3d * dt;
      F_x.block<3, 3>(7, 0) = -R_imu * acc_avr_skew * dt;
      if (ba_bg_est_en) F_x.block<3, 3>(7, 13) = -R_imu * dt;
      if (gravity_est_en) F_x.block<3, 3>(7, 16) = Eye3d * dt;

      // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
      // F_x(6,6) = 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) * (-tau*tau); F_x(18,18) = 0.00001;
      if (exposure_estimate_en) cov_w(6, 6) = cov_inv_expo * dt * dt;
      cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
      cov_w.block<3, 3>(7, 7) = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
      cov_w.block<3, 3>(10, 10).diagonal() = cov_bias_gyr * dt * dt; // bias gyro covariance
      cov_w.block<3, 3>(13, 13).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

      state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
      // state_inout.cov.block<18,18>(0,0) = F_x.block<18,18>(0,0) *
      // state_inout.cov.block<18,18>(0,0) * F_x.block<18,18>(0,0).transpose() +
      // cov_w.block<18,18>(0,0);

      // tau = tau + 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) *
      // (-tau*tau) * dt;

      // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);

      /* propogation of IMU attitude */
      R_imu = R_imu * Exp_f;

      /* Specific acceleration (global frame) of IMU */
      acc_imu = R_imu * acc_avr + state_inout.gravity;

      /* propogation of IMU */
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

      /* velocity of IMU */
      vel_imu = vel_imu + acc_imu * dt;

      /* save the poses at each IMU measurements */
      angvel_last = angvel_avr;
      acc_s_last = acc_imu;

      // cout<<setw(20)<<"offset_t: "<<offs_t<<"tail->header.stamp.toSec():
      // "<<tail->header.stamp.toSec()<<endl; printf("[ LIO Propagation ]
      // offs_t: %lf \n", offs_t);
      IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    }

    // unbiased_gyr = V3D(IMUpose.back().gyr[0], IMUpose.back().gyr[1], IMUpose.back().gyr[2]);
    // cout<<"prop end - start: "<<prop_end_time - prop_beg_time<<" dt_all: "<<dt_all<<endl;
    lidar_meas.last_lio_update_time = prop_end_time;
    // dt = prop_end_time - imu_end_time;
    // printf("[ LIO Propagation ] dt: %lf \n", dt);
    break;
  }

  state_inout.vel_end = vel_imu;
  state_inout.rot_end = R_imu;
  state_inout.pos_end = pos_imu;
  state_inout.inv_expo_time = tau;

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // if (imu_end_time>prop_beg_time)
  // {
  //   double note = prop_end_time > imu_end_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - imu_end_time);
  //   state_inout.vel_end = vel_imu + note * acc_imu * dt;
  //   state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }
  // else
  // {
  //   double note = prop_end_time > prop_beg_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - prop_beg_time);
  //   state_inout.vel_end = vel_imu + note * acc_imu * dt;
  //   state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }

  // cout<<"[ Propagation ] output state: "<<state_inout.vel_end.transpose() <<
  // state_inout.pos_end.transpose()<<endl;

  last_imu = v_imu.back();
  last_prop_end_time = prop_end_time;

  double t1 = omp_get_wtime();

  // auto pos_liD_e = state_inout.pos_end + state_inout.rot_end *
  // Lid_offset_to_IMU; auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

  // cout<<"[ IMU ]: vel "<<state_inout.vel_end.transpose()<<" pos
  // "<<state_inout.pos_end.transpose()<<"
  // ba"<<state_inout.bias_a.transpose()<<" bg
  // "<<state_inout.bias_g.transpose()<<endl; cout<<"propagated cov:
  // "<<state_inout.cov.diagonal().transpose()<<endl;

  //   cout<<"UndistortPcl Time:";
  //   for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
  //     cout<<it->offset_time<<" ";
  //   }
  //   cout<<endl<<"UndistortPcl size:"<<IMUpose.size()<<endl;
  //   cout<<"Undistorted pcl_out.size: "<<pcl_out.size()
  //          <<"lidar_meas.size: "<<lidar_meas.lidar->points.size()<<endl;
  if (pcl_wait_proc.points.size() < 1) return;

  /*** undistort each lidar point (backward propagation), ONLY working for LIO
   * update ***/
  if (lidar_meas.lio_vio_flg == LIO)
  {
    auto it_pcl = pcl_wait_proc.points.end() - 1;
    M3D extR_Ri(Lid_rot_to_IMU.transpose() * state_inout.rot_end.transpose());
    V3D exrR_extT(Lid_rot_to_IMU.transpose() * Lid_offset_to_IMU);
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
      auto head = it_kp - 1;
      auto tail = it_kp;
      R_imu << MAT_FROM_ARRAY(head->rot);
      acc_imu << VEC_FROM_ARRAY(head->acc);
      // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      angvel_avr << VEC_FROM_ARRAY(head->gyr);

      // printf("head->offset_time: %lf \n", head->offset_time);
      // printf("it_pcl->curvature: %lf pt dt: %lf \n", it_pcl->curvature,
      // it_pcl->curvature / double(1000) - head->offset_time);

      for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
      {
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        /* Transform to the 'end' frame */
        M3D R_i(R_imu * Exp(angvel_avr, dt));
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - state_inout.pos_end);

        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        // V3D P_compensate = Lid_rot_to_IMU.transpose() *
        // (state_inout.rot_end.transpose() * (R_i * (Lid_rot_to_IMU * P_i +
        // Lid_offset_to_IMU) + T_ei) - Lid_offset_to_IMU);
        V3D P_compensate = (extR_Ri * (R_i * (Lid_rot_to_IMU * P_i + Lid_offset_to_IMU) + T_ei) - exrR_extT);

        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == pcl_wait_proc.points.begin()) break;
      }
    }
    pcl_out = pcl_wait_proc;
    pcl_wait_proc.clear();
    IMUpose.clear();
  }
  // printf("[ IMU ] time forward: %lf, backward: %lf.\n", t1 - t0, omp_get_wtime() - t1);
}

void ImuProcess::Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1, t2, t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);
  if (!imu_en)
  {
    Forward_without_imu(lidar_meas, stat, *cur_pcl_un_);
    return;
  }

  MeasureGroup meas = lidar_meas.measures.back();

  if (imu_need_init)
  {
    double pcl_end_time = lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;
    // lidar_meas.last_lio_update_time = pcl_end_time;

    if (meas.imu.empty()) { return; };
    /// The very first lidar frame
    IMU_init(meas, stat, init_iter_num);

    imu_need_init = true;

    last_imu = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT)
    {
      // cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; acc covarience: "
               "%.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f \n",
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(), cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1],
               cov_gyr[2]);
      ROS_INFO("IMU Initials: ba covarience: %.8f %.8f %.8f; bg covarience: "
               "%.8f %.8f %.8f",
               cov_bias_acc[0], cov_bias_acc[1], cov_bias_acc[2], cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
    }

    return;
  }

  UndistortPcl(lidar_meas, stat, *cur_pcl_un_);
  // cout << "[ IMU ] undistorted point num: " << cur_pcl_un_->size() << endl;
}
