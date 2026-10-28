#ifndef FAST_LIVO_RAW_LIDAR_REPROCESS_H
#define FAST_LIVO_RAW_LIDAR_REPROCESS_H

#include "common_lib.h"

#include <livox_ros_driver/CustomMsg.h>
#include <ros/serialization.h>
#include <sensor_msgs/PointCloud2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace fast_livo
{
namespace raw_replay
{

inline std::uint64_t fnv1a(const void *data, const std::size_t size)
{
  std::uint64_t value = 1469598103934665603ULL;
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  for (std::size_t i = 0; i < size; ++i)
  {
    value ^= bytes[i];
    value *= 1099511628211ULL;
  }
  return value;
}

template <typename Message>
RawLidarFrame capture(const Message &message, const RawLidarMessageType type,
                      const double effective_header_time,
                      const PointCloudXYZI::Ptr &processed,
                      const double oracle_source_relative_time,
                      const double oracle_inverse_range_scale,
                      const bool oracle_range_scale_active)
{
  RawLidarFrame frame;
  frame.message_type = type;
  frame.sensor_header_time = message.header.stamp.toSec();
  frame.effective_header_time = effective_header_time;
  frame.oracle_source_relative_time = oracle_source_relative_time;
  frame.oracle_inverse_range_scale = oracle_inverse_range_scale;
  frame.oracle_range_scale_active = oracle_range_scale_active;
  if (!std::isfinite(frame.sensor_header_time) ||
      !std::isfinite(frame.effective_header_time) ||
      !std::isfinite(frame.oracle_source_relative_time) ||
      !std::isfinite(frame.oracle_inverse_range_scale) ||
      frame.oracle_inverse_range_scale <= 0.0)
    throw std::invalid_argument("invalid raw LiDAR frame metadata");

  if (processed && !processed->empty())
  {
    frame.minimum_processed_offset_ms =
        std::numeric_limits<float>::infinity();
    frame.maximum_processed_offset_ms =
        -std::numeric_limits<float>::infinity();
    for (const PointType &point : processed->points)
    {
      frame.minimum_processed_offset_ms = std::min(
          frame.minimum_processed_offset_ms, point.curvature);
      frame.maximum_processed_offset_ms = std::max(
          frame.maximum_processed_offset_ms, point.curvature);
    }
  }

  const std::uint32_t size = ros::serialization::serializationLength(message);
  frame.serialized_message.resize(size);
  if (size != 0)
  {
    ros::serialization::OStream stream(frame.serialized_message.data(), size);
    ros::serialization::serialize(stream, message);
  }
  frame.serialized_checksum = fnv1a(
      frame.serialized_message.data(), frame.serialized_message.size());
  return frame;
}

template <typename Message>
Message decode(const RawLidarFrame &frame)
{
  if (fnv1a(frame.serialized_message.data(), frame.serialized_message.size()) !=
      frame.serialized_checksum)
    throw std::runtime_error("raw LiDAR payload checksum mismatch");
  Message message;
  if (!frame.serialized_message.empty())
  {
    // ROS deserialization does not mutate the byte buffer.
    auto *bytes = const_cast<std::uint8_t *>(frame.serialized_message.data());
    ros::serialization::IStream stream(bytes, frame.serialized_message.size());
    ros::serialization::deserialize(stream, message);
  }
  return message;
}

inline bool hostIsBigEndian()
{
  const std::uint16_t value = 0x0102;
  return *reinterpret_cast<const std::uint8_t *>(&value) == 0x01;
}

inline const sensor_msgs::PointField *findField(
    const sensor_msgs::PointCloud2 &message, const char *name)
{
  for (const sensor_msgs::PointField &field : message.fields)
    if (field.name == name) return &field;
  return nullptr;
}

template <typename T>
T byteSwap(T value)
{
  static_assert(std::is_trivially_copyable<T>::value,
                "byteSwap requires a trivially copyable value");
  T output;
  const auto *input = reinterpret_cast<const std::uint8_t *>(&value);
  auto *result = reinterpret_cast<std::uint8_t *>(&output);
  std::reverse_copy(input, input + sizeof(T), result);
  return output;
}

template <typename T>
T readScalar(const std::uint8_t *point, const sensor_msgs::PointField &field,
             const bool swap)
{
  T value;
  std::memcpy(&value, point + field.offset, sizeof(T));
  return swap ? byteSwap(value) : value;
}

template <typename T>
void writeScalar(std::uint8_t *point, const sensor_msgs::PointField &field,
                 T value, const bool swap)
{
  if (swap) value = byteSwap(value);
  std::memcpy(point + field.offset, &value, sizeof(T));
}

inline double readCoordinate(const std::uint8_t *point,
                             const sensor_msgs::PointField &field,
                             const bool swap)
{
  if (field.datatype == sensor_msgs::PointField::FLOAT32)
    return readScalar<float>(point, field, swap);
  if (field.datatype == sensor_msgs::PointField::FLOAT64)
    return readScalar<double>(point, field, swap);
  throw std::runtime_error("raw LiDAR coordinate must be FLOAT32 or FLOAT64");
}

inline void writeCoordinate(std::uint8_t *point,
                            const sensor_msgs::PointField &field,
                            const bool swap, const double value)
{
  if (field.datatype == sensor_msgs::PointField::FLOAT32)
    writeScalar<float>(point, field, static_cast<float>(value), swap);
  else if (field.datatype == sensor_msgs::PointField::FLOAT64)
    writeScalar<double>(point, field, value, swap);
  else
    throw std::runtime_error("raw LiDAR coordinate must be FLOAT32 or FLOAT64");
}

inline std::size_t applyRangeScale(sensor_msgs::PointCloud2 &message,
                                   const double inverse_scale)
{
  if (!std::isfinite(inverse_scale) || inverse_scale <= 0.0)
    throw std::invalid_argument("raw LiDAR inverse scale must be positive");
  // Identity correction must be a true no-op: do not rewrite serialized
  // FLOAT32/FLOAT64 coordinates and do not report unchanged points as fixed.
  if (inverse_scale == 1.0) return 0;
  const auto *x = findField(message, "x");
  const auto *y = findField(message, "y");
  const auto *z = findField(message, "z");
  if (!x || !y || !z || message.point_step == 0)
    throw std::runtime_error("raw PointCloud2 lacks x/y/z fields");
  const bool swap = message.is_bigendian != hostIsBigEndian();
  std::size_t corrected = 0;
  for (std::size_t row = 0; row < message.height; ++row)
  {
    std::uint8_t *row_begin = message.data.data() + row * message.row_step;
    for (std::size_t column = 0; column < message.width; ++column)
    {
      std::uint8_t *point = row_begin + column * message.point_step;
      if (point + message.point_step > message.data.data() + message.data.size())
        throw std::runtime_error("raw PointCloud2 row layout exceeds payload");
      const double px = readCoordinate(point, *x, swap);
      const double py = readCoordinate(point, *y, swap);
      const double pz = readCoordinate(point, *z, swap);
      if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz))
        continue;
      writeCoordinate(point, *x, swap, px * inverse_scale);
      writeCoordinate(point, *y, swap, py * inverse_scale);
      writeCoordinate(point, *z, swap, pz * inverse_scale);
      ++corrected;
    }
  }
  return corrected;
}

inline std::size_t applyRangeScale(livox_ros_driver::CustomMsg &message,
                                   const double inverse_scale)
{
  if (!std::isfinite(inverse_scale) || inverse_scale <= 0.0)
    throw std::invalid_argument("raw LiDAR inverse scale must be positive");
  if (inverse_scale == 1.0) return 0;
  std::size_t corrected = 0;
  for (auto &point : message.points)
  {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z)) continue;
    point.x = static_cast<float>(point.x * inverse_scale);
    point.y = static_cast<float>(point.y * inverse_scale);
    point.z = static_cast<float>(point.z * inverse_scale);
    ++corrected;
  }
  return corrected;
}

inline bool contributesAfter(const RawLidarFrame &frame,
                             const double boundary_time)
{
  return frame.effective_header_time +
      static_cast<double>(frame.maximum_processed_offset_ms) / 1000.0 >=
      boundary_time;
}

// Keep replay's time rebasing bit-for-bit aligned with sync_packages().  The
// float point time must be promoted for the double addition; rounding the
// offset to float first can change the final curvature by one ULP.
inline float rebaseCurvatureLikeOnline(float curvature,
                                       const double frame_header_time,
                                       const double event_begin_time)
{
  curvature += (frame_header_time - event_begin_time) * 1000.0f;
  return curvature;
}

} // namespace raw_replay
} // namespace fast_livo

#endif // FAST_LIVO_RAW_LIDAR_REPROCESS_H
