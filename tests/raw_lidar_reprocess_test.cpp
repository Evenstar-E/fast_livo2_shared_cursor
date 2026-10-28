#include "preprocess.h"
#include "raw_lidar_reprocess.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace
{
sensor_msgs::PointCloud2 makeXt32(const float scale)
{
  sensor_msgs::PointCloud2 message;
  message.header.stamp = ros::Time(10.0);
  message.header.frame_id = "lidar";
  sensor_msgs::PointCloud2Modifier modifier(message);
  modifier.setPointCloud2Fields(
      6,
      "x", 1, sensor_msgs::PointField::FLOAT32,
      "y", 1, sensor_msgs::PointField::FLOAT32,
      "z", 1, sensor_msgs::PointField::FLOAT32,
      "intensity", 1, sensor_msgs::PointField::FLOAT32,
      "timestamp", 1, sensor_msgs::PointField::FLOAT64,
      "ring", 1, sensor_msgs::PointField::UINT16);
  modifier.resize(3);
  sensor_msgs::PointCloud2Iterator<float> x(message, "x");
  sensor_msgs::PointCloud2Iterator<float> y(message, "y");
  sensor_msgs::PointCloud2Iterator<float> z(message, "z");
  sensor_msgs::PointCloud2Iterator<float> intensity(message, "intensity");
  sensor_msgs::PointCloud2Iterator<double> timestamp(message, "timestamp");
  sensor_msgs::PointCloud2Iterator<std::uint16_t> ring(message, "ring");
  const float ranges[] = {0.49f, 1.0f, 2.0f};
  for (int i = 0; i < 3; ++i, ++x, ++y, ++z, ++intensity, ++timestamp, ++ring)
  {
    *x = ranges[i] * scale;
    *y = 0.0f;
    *z = 0.0f;
    *intensity = static_cast<float>(10 + i);
    *timestamp = 10.0 + i * 0.001;
    *ring = static_cast<std::uint16_t>(i);
  }
  return message;
}

PointCloudXYZI::Ptr process(const sensor_msgs::PointCloud2 &message)
{
  Preprocess preprocessor;
  preprocessor.set(false, XT32, 0.5, 1);
  preprocessor.N_SCANS = 32;
  preprocessor.blind_sqr = 0.25;
  PointCloudXYZI::Ptr output(new PointCloudXYZI());
  sensor_msgs::PointCloud2::ConstPtr input(
      new sensor_msgs::PointCloud2(message));
  preprocessor.process(input, output);
  return output;
}
} // namespace

int main()
{
  const sensor_msgs::PointCloud2 clean = makeXt32(1.0f);
  const PointCloudXYZI::Ptr clean_processed = process(clean);
  assert(clean_processed->size() == 2);

  const RawLidarFrame frame = fast_livo::raw_replay::capture(
      clean, RawLidarMessageType::POINTCLOUD2, 10.0, clean_processed,
      0.0, 1.0, false);
  sensor_msgs::PointCloud2 decoded =
      fast_livo::raw_replay::decode<sensor_msgs::PointCloud2>(frame);
  const PointCloudXYZI::Ptr replayed = process(decoded);
  assert(replayed->size() == clean_processed->size());
  for (std::size_t i = 0; i < replayed->size(); ++i)
  {
    const PointType &a = replayed->points[i];
    const PointType &b = clean_processed->points[i];
    assert(a.x == b.x && a.y == b.y && a.z == b.z);
    assert(a.intensity == b.intensity && a.curvature == b.curvature);
  }

  sensor_msgs::PointCloud2 identity = clean;
  const RawLidarFrame identity_before = fast_livo::raw_replay::capture(
      identity, RawLidarMessageType::POINTCLOUD2, 10.0, clean_processed,
      0.0, 1.0, true);
  assert(fast_livo::raw_replay::applyRangeScale(identity, 1.0) == 0);
  const RawLidarFrame identity_after = fast_livo::raw_replay::capture(
      identity, RawLidarMessageType::POINTCLOUD2, 10.0, clean_processed,
      0.0, 1.0, true);
  assert(identity_before.serialized_checksum ==
         identity_after.serialized_checksum);

  sensor_msgs::PointCloud2 faulted = makeXt32(1.04f);
  const PointCloudXYZI::Ptr faulted_processed = process(faulted);
  assert(faulted_processed->size() == 3); // blind-gate membership changed
  const std::size_t corrected =
      fast_livo::raw_replay::applyRangeScale(faulted, 1.0 / 1.04);
  assert(corrected == 3);
  const PointCloudXYZI::Ptr raw_corrected = process(faulted);
  assert(raw_corrected->size() == 2); // reprocessing restores the clean gate
  assert(raw_corrected->points[0].intensity == clean_processed->points[0].intensity);
  assert(raw_corrected->points[1].intensity == clean_processed->points[1].intensity);

  RawLidarFrame corrupted = frame;
  corrupted.serialized_message.front() ^= 0x1;
  bool checksum_rejected = false;
  try
  {
    (void)fast_livo::raw_replay::decode<sensor_msgs::PointCloud2>(corrupted);
  }
  catch (const std::runtime_error &)
  {
    checksum_rejected = true;
  }
  assert(checksum_rejected);
  assert(fast_livo::raw_replay::contributesAfter(frame, 10.001));
  assert(!fast_livo::raw_replay::contributesAfter(frame, 10.003));

  // Regression for the first raw-noop integration divergence.  Match the
  // online mixed float/double expression exactly; pre-rounding the offset to
  // float is observably different for some legitimate timestamps.
  bool found_pre_rounded_difference = false;
  for (int i = 1; i <= 100000; ++i)
  {
    const float source_curvature = static_cast<float>((i % 1000) * 0.001);
    const double begin_time = 78.423429727554321;
    const double frame_time = begin_time + static_cast<double>(i) * 1e-9;
    float online = source_curvature;
    online += (frame_time - begin_time) * 1000.0f;
    const float rebuilt = fast_livo::raw_replay::rebaseCurvatureLikeOnline(
        source_curvature, frame_time, begin_time);
    assert(rebuilt == online);

    float pre_rounded = source_curvature;
    pre_rounded += static_cast<float>((frame_time - begin_time) * 1000.0);
    if (pre_rounded != online) found_pre_rounded_difference = true;
  }
  assert(found_pre_rounded_difference);
  return 0;
}
