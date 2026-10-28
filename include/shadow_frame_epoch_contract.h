#pragma once

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace fast_livo
{
namespace frame_epoch
{

// The candidate-native map is immutable in its own frame.  The public
// presentation frame may move relative to it during the bounded pose bridge.
// These copies are diagnostic sidecars; canonical mux topics are unchanged.
inline void requireFrames(const std::string &source_frame,
                          const std::string &presentation_frame,
                          const std::string &native_frame)
{
  if (presentation_frame.empty() || native_frame.empty() ||
      presentation_frame == native_frame || source_frame != presentation_frame)
    throw std::invalid_argument("invalid frame epoch source or destination");
}

inline nav_msgs::Odometry nativeOdometry(const nav_msgs::Odometry &source,
                                         const std::string &presentation_frame,
                                         const std::string &native_frame)
{
  requireFrames(source.header.frame_id, presentation_frame, native_frame);
  nav_msgs::Odometry output = source;
  output.header.frame_id = native_frame;
  return output;
}

inline nav_msgs::Path nativePath(const nav_msgs::Path &source,
                                const std::string &presentation_frame,
                                const std::string &native_frame)
{
  requireFrames(source.header.frame_id, presentation_frame, native_frame);
  nav_msgs::Path output = source;
  output.header.frame_id = native_frame;
  for (auto &stamped : output.poses)
  {
    if (!stamped.header.frame_id.empty() &&
        stamped.header.frame_id != presentation_frame)
      throw std::invalid_argument("path contains a different source frame");
    stamped.header.frame_id = native_frame;
  }
  return output;
}

inline sensor_msgs::PointCloud2 nativeCloud(const sensor_msgs::PointCloud2 &source,
                                           const std::string &presentation_frame,
                                           const std::string &native_frame)
{
  requireFrames(source.header.frame_id, presentation_frame, native_frame);
  sensor_msgs::PointCloud2 output = source;
  output.header.frame_id = native_frame;
  // In particular, output.data and every PointField remain byte-for-byte
  // identical.  This holds for both registered clouds and published maps.
  return output;
}

struct Translation
{
  double x = 0.0, y = 0.0, z = 0.0;
};

// A preview is a single presentation-frame snapshot, not a replacement for
// the timestamped canonical path. Every historical pose uses the transform
// at snapshot_stamp; its individual measurement stamp remains unchanged.
inline nav_msgs::Path snapshotPath(const nav_msgs::Path &source,
                                   const std::string &presentation_frame,
                                   const ros::Time &snapshot_stamp,
                                   const Translation &translation)
{
  if (source.header.frame_id != presentation_frame ||
      presentation_frame.empty() || snapshot_stamp.isZero() ||
      !std::isfinite(translation.x) || !std::isfinite(translation.y) ||
      !std::isfinite(translation.z) ||
      source.header.stamp > snapshot_stamp)
    throw std::invalid_argument("invalid frame epoch snapshot path");
  nav_msgs::Path output = source;
  output.header.stamp = snapshot_stamp;
  for (auto &stamped : output.poses)
  {
    if (!stamped.header.frame_id.empty() &&
        stamped.header.frame_id != presentation_frame)
      throw std::invalid_argument("snapshot path contains a different frame");
    stamped.header.frame_id = presentation_frame;
    stamped.pose.position.x += translation.x;
    stamped.pose.position.y += translation.y;
    stamped.pose.position.z += translation.z;
  }
  return output;
}

inline Translation presentationFromNative(
    const nav_msgs::Odometry &native,
    const nav_msgs::Odometry &presentation)
{
  if (native.header.stamp != presentation.header.stamp ||
      native.child_frame_id != presentation.child_frame_id)
    throw std::invalid_argument("frame epoch odometry pair does not match");
  Translation value;
  value.x = presentation.pose.pose.position.x - native.pose.pose.position.x;
  value.y = presentation.pose.pose.position.y - native.pose.pose.position.y;
  value.z = presentation.pose.pose.position.z - native.pose.pose.position.z;
  if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
    throw std::invalid_argument("non-finite frame epoch translation");
  return value;
}

} // namespace frame_epoch
} // namespace fast_livo
