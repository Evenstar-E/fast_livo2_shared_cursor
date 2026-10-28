#pragma once
#include "common_lib.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

// Diagnostic only. Hash semantic scalars, never PCL/Eigen padding, cv::Mat
// pointers, or unused normals. FNV identifies first differences, not security.
namespace fast_livo { namespace audit {
struct Digest {
  std::uint64_t value = 14695981039346656037ULL;
  void bytes(const void *p, std::size_t n) {
    const auto *b = static_cast<const unsigned char *>(p);
    for (std::size_t i=0; i<n; ++i) { value ^= b[i]; value *= 1099511628211ULL; }
  }
  template<class T> void scalar(const T &v) { bytes(&v, sizeof(v)); }
  std::string hex() const { std::ostringstream s; s << std::hex << value; return s.str(); }
};
inline std::string cloudDigest(const PointCloudXYZI::Ptr &cloud) {
  Digest d; std::uint64_t n = cloud ? cloud->size() : 0; d.scalar(n);
  if (cloud) for (const auto &p : cloud->points) {
    d.scalar(p.x); d.scalar(p.y); d.scalar(p.z); d.scalar(p.intensity); d.scalar(p.curvature);
  }
  return d.hex();
}
struct InputDigest {
  std::string cur_hash;
  std::string image_hash;
  std::string imu_hash;
};
inline InputDigest inputDigest(const LidarMeasureGroup &m) {
  InputDigest out;
  out.cur_hash = cloudDigest(m.pcl_proc_cur);
  Digest images, imus;
  for (const auto &g : m.measures) {
    images.scalar(g.vio_time); images.scalar(g.img.rows); images.scalar(g.img.cols);
    const int type = g.img.type(); images.scalar(type);
    for (int r=0;r<g.img.rows;++r) images.bytes(g.img.ptr(r), g.img.cols*g.img.elemSize());
    imus.scalar(g.lio_time); std::uint64_t n=g.imu.size(); imus.scalar(n);
    for (const auto &imu : g.imu) {
      imus.scalar(imu->header.stamp.sec); imus.scalar(imu->header.stamp.nsec);
      imus.scalar(imu->linear_acceleration.x); imus.scalar(imu->linear_acceleration.y); imus.scalar(imu->linear_acceleration.z);
      imus.scalar(imu->angular_velocity.x); imus.scalar(imu->angular_velocity.y); imus.scalar(imu->angular_velocity.z);
      imus.scalar(imu->orientation.x); imus.scalar(imu->orientation.y); imus.scalar(imu->orientation.z); imus.scalar(imu->orientation.w);
      for (double x:imu->linear_acceleration_covariance) imus.scalar(x);
      for (double x:imu->angular_velocity_covariance) imus.scalar(x);
      for (double x:imu->orientation_covariance) imus.scalar(x);
    }
  }
  out.image_hash = images.hex();
  out.imu_hash = imus.hex();
  return out;
}
class FusionEventAudit {
  std::ofstream stream_;
public:
  void open(const std::string &path) {
    if (path.empty()) return;
    stream_.open(path, std::ios::out | std::ios::trunc);
    if (!stream_) throw std::runtime_error("Cannot open fusion audit: " + path);
    stream_ << "event,route,stage,type,relative_time,lio_time,vio_time,lidar_begin,lidar_end,last_lio_time,cur_count,next_count,cur_hash,next_hash,lidar_hash,image_hash,imu_hash,imu_stamps,state_hash,state_values,undistorted_hash\n";
    stream_ << std::setprecision(17);
  }
  bool enabled() const { return stream_.is_open(); }
  void write(std::uint64_t event, const std::string &route, const char *stage,
             const LidarMeasureGroup &m, double origin, const StatesGroup &state,
             const PointCloudXYZI::Ptr &undistorted) {
    if (!enabled() || m.measures.empty()) return;
    const InputDigest input = inputDigest(m);
    Digest states;
    std::ostringstream stamps, values; stamps << std::setprecision(17); values << std::setprecision(17);
    auto stateValue = [&](double x) { states.scalar(x); values << x << ';'; };
    for (int i=0;i<3;++i) stateValue(state.pos_end[i]);
    for (int r=0;r<3;++r) for(int c=0;c<3;++c) stateValue(state.rot_end(r,c));
    for (int i=0;i<3;++i) stateValue(state.vel_end[i]);
    for (int i=0;i<3;++i) stateValue(state.bias_g[i]);
    for (int i=0;i<3;++i) stateValue(state.bias_a[i]);
    for (int i=0;i<3;++i) stateValue(state.gravity[i]);
    stateValue(state.inv_expo_time);
    for (int r=0;r<state.cov.rows();++r) for(int c=0;c<state.cov.cols();++c) stateValue(state.cov(r,c));
    for (const auto &g : m.measures) {
      for (const auto &imu : g.imu)
        stamps << imu->header.stamp.toNSec() << ';';
    }
    const auto &g=m.measures.back();
    stream_ << event << ',' << route << ',' << stage << ',' << static_cast<int>(m.lio_vio_flg)
      << ',' << (m.lio_vio_flg==VIO?g.vio_time:g.lio_time)-origin << ',' << g.lio_time << ',' << g.vio_time
      << ',' << m.lidar_frame_beg_time << ',' << m.lidar_frame_end_time << ',' << m.last_lio_update_time
      << ',' << m.pcl_proc_cur->size() << ',' << m.pcl_proc_next->size()
      << ',' << input.cur_hash << ',' << cloudDigest(m.pcl_proc_next) << ',' << cloudDigest(m.lidar)
      << ',' << input.image_hash << ',' << input.imu_hash << ',' << stamps.str()
      << ',' << states.hex() << ',' << values.str() << ',' << cloudDigest(undistorted) << '\n';
    stream_.flush(); // main exits with _Exit; preserve the last completed stage.
  }
};
}} // namespace
