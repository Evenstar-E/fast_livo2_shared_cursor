#include "fault_injection_model.h"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace fi = fast_livo::fault_injection;

namespace
{

const sensor_msgs::PointField *findField(const sensor_msgs::PointCloud2 &msg,
                                         const std::string &name)
{
  for (const auto &field : msg.fields)
    if (field.name == name) return &field;
  return nullptr;
}

template <typename T>
T byteSwap(T value)
{
  std::uint8_t bytes[sizeof(T)];
  std::memcpy(bytes, &value, sizeof(T));
  std::reverse(bytes, bytes + sizeof(T));
  std::memcpy(&value, bytes, sizeof(T));
  return value;
}

bool hostIsBigEndian()
{
  const std::uint16_t value = 0x0102;
  return *reinterpret_cast<const std::uint8_t *>(&value) == 0x01;
}

bool readNumeric(const std::uint8_t *point, const sensor_msgs::PointField &field,
                 bool swap_endian, double &value)
{
  if (field.datatype == sensor_msgs::PointField::FLOAT32)
  {
    float raw;
    std::memcpy(&raw, point + field.offset, sizeof(raw));
    if (swap_endian) raw = byteSwap(raw);
    value = raw;
    return true;
  }
  if (field.datatype == sensor_msgs::PointField::FLOAT64)
  {
    double raw;
    std::memcpy(&raw, point + field.offset, sizeof(raw));
    if (swap_endian) raw = byteSwap(raw);
    value = raw;
    return true;
  }
  return false;
}

bool writeNumeric(std::uint8_t *point, const sensor_msgs::PointField &field,
                  bool swap_endian, double value)
{
  if (field.datatype == sensor_msgs::PointField::FLOAT32)
  {
    float raw = static_cast<float>(value);
    if (swap_endian) raw = byteSwap(raw);
    std::memcpy(point + field.offset, &raw, sizeof(raw));
    return true;
  }
  if (field.datatype == sensor_msgs::PointField::FLOAT64)
  {
    double raw = value;
    if (swap_endian) raw = byteSwap(raw);
    std::memcpy(point + field.offset, &raw, sizeof(raw));
    return true;
  }
  return false;
}

std::string escapeJson(const std::string &value)
{
  std::ostringstream out;
  for (const char ch : value)
  {
    if (ch == '\\' || ch == '"') out << '\\';
    out << ch;
  }
  return out.str();
}

} // namespace

class PointCloudFaultInjector
{
public:
  PointCloudFaultInjector() : private_nh_("~"), model_(loadConfig())
  {
    private_nh_.param<std::string>("input_topic", input_topic_, "/hesai/pandar");
    private_nh_.param<std::string>("output_topic", output_topic_, "/fault_injection/hesai_pandar");

    output_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 16);
    active_pub_ = private_nh_.advertise<std_msgs::Bool>("active", 1, true);
    metadata_pub_ = private_nh_.advertise<std_msgs::String>("metadata", 1, true);
    input_sub_ = nh_.subscribe(input_topic_, 16, &PointCloudFaultInjector::callback, this,
                               ros::TransportHints().tcpNoDelay());

    publishActive(false);
    std_msgs::String metadata;
    metadata.data = metadataJson();
    metadata_pub_.publish(metadata);
    ROS_WARN_STREAM("[ FaultInjector ] " << metadata.data);
  }

private:
  fi::Config loadConfig()
  {
    fi::Config config;
    std::string mode = fi::modeName(config.mode);
    int seed = static_cast<int>(config.seed);
    private_nh_.param<std::string>("mode", mode, mode);
    private_nh_.param<double>("start_sec", config.start_sec, config.start_sec);
    private_nh_.param<double>("end_sec", config.end_sec, config.end_sec);
    private_nh_.param<double>("range_scale_start", config.range_scale_start, config.range_scale_start);
    private_nh_.param<double>("range_scale_end", config.range_scale_end, config.range_scale_end);
    private_nh_.param<double>("ghost_fraction", config.ghost_fraction, config.ghost_fraction);
    private_nh_.param<double>("ghost_yaw_deg", config.ghost_yaw_deg, config.ghost_yaw_deg);
    private_nh_.param<double>("ghost_translation_x", config.ghost_translation_x, config.ghost_translation_x);
    private_nh_.param<double>("ghost_translation_y", config.ghost_translation_y, config.ghost_translation_y);
    private_nh_.param<double>("ghost_translation_z", config.ghost_translation_z, config.ghost_translation_z);
    private_nh_.param<double>("point_time_scale", config.point_time_scale, config.point_time_scale);
    private_nh_.param<int>("seed", seed, seed);
    config.seed = static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed));
    config.mode = fi::parseMode(mode);
    return config;
  }

  std::string metadataJson() const
  {
    const auto &config = model_.config();
    std::ostringstream out;
    out << std::setprecision(12)
        << "{\"schema\":1,\"input_topic\":\"" << escapeJson(input_topic_)
        << "\",\"output_topic\":\"" << escapeJson(output_topic_)
        << "\",\"mode\":\"" << fi::modeName(config.mode)
        << "\",\"start_sec\":" << config.start_sec
        << ",\"end_sec\":" << config.end_sec
        << ",\"range_scale_start\":" << config.range_scale_start
        << ",\"range_scale_end\":" << config.range_scale_end
        << ",\"ghost_fraction\":" << config.ghost_fraction
        << ",\"ghost_yaw_deg\":" << config.ghost_yaw_deg
        << ",\"ghost_translation\":[" << config.ghost_translation_x << ','
        << config.ghost_translation_y << ',' << config.ghost_translation_z << ']'
        << ",\"point_time_scale\":" << config.point_time_scale
        << ",\"seed\":" << config.seed << '}';
    return out.str();
  }

  void publishActive(bool active)
  {
    if (active == active_) return;
    active_ = active;
    std_msgs::Bool message;
    message.data = active;
    active_pub_.publish(message);
    ROS_WARN_STREAM("[ FaultInjector ] fault window " << (active ? "ACTIVE" : "INACTIVE"));
  }

  void callback(const sensor_msgs::PointCloud2::ConstPtr &input)
  {
    if (!epoch_initialized_)
    {
      epoch_stamp_ = input->header.stamp;
      epoch_initialized_ = true;
      ROS_INFO_STREAM("[ FaultInjector ] epoch=" << std::fixed << std::setprecision(9)
                      << epoch_stamp_.toSec());
    }

    const double relative_sec = (input->header.stamp - epoch_stamp_).toSec();
    const bool active = model_.active(relative_sec);
    publishActive(active);

    if (!active)
    {
      output_pub_.publish(input);
      ++scan_index_;
      return;
    }

    sensor_msgs::PointCloud2 output = *input;
    const auto *x_field = findField(output, "x");
    const auto *y_field = findField(output, "y");
    const auto *z_field = findField(output, "z");
    if (!x_field || !y_field || !z_field || output.point_step == 0)
    {
      ROS_ERROR_THROTTLE(1.0, "[ FaultInjector ] PointCloud2 lacks usable x/y/z fields");
      output_pub_.publish(output);
      ++scan_index_;
      return;
    }

    const bool swap_endian = output.is_bigendian != hostIsBigEndian();
    const auto *timestamp_field = findField(output, "timestamp");
    double timestamp_origin = 0.0;
    bool scale_point_time = timestamp_field && model_.config().point_time_scale != 1.0 &&
                            !output.data.empty() &&
                            readNumeric(output.data.data(), *timestamp_field, swap_endian, timestamp_origin);

    std::size_t corrupted_points = 0;
    std::size_t index = 0;
    for (std::size_t row = 0; row < output.height; ++row)
    {
      std::uint8_t *row_begin = output.data.data() + row * output.row_step;
      for (std::size_t column = 0; column < output.width; ++column, ++index)
      {
        std::uint8_t *point = row_begin + column * output.point_step;
        if (point + output.point_step > output.data.data() + output.data.size()) break;

        double x, y, z;
        if (!readNumeric(point, *x_field, swap_endian, x) ||
            !readNumeric(point, *y_field, swap_endian, y) ||
            !readNumeric(point, *z_field, swap_endian, z))
        {
          ROS_ERROR_THROTTLE(1.0, "[ FaultInjector ] x/y/z must be FLOAT32 or FLOAT64");
          break;
        }
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

        model_.corruptPoint(relative_sec, index, scan_index_, x, y, z);
        writeNumeric(point, *x_field, swap_endian, x);
        writeNumeric(point, *y_field, swap_endian, y);
        writeNumeric(point, *z_field, swap_endian, z);
        ++corrupted_points;

        if (scale_point_time)
        {
          double timestamp;
          if (readNumeric(point, *timestamp_field, swap_endian, timestamp))
          {
            timestamp = timestamp_origin +
                        (timestamp - timestamp_origin) * model_.config().point_time_scale;
            writeNumeric(point, *timestamp_field, swap_endian, timestamp);
          }
        }
      }
    }

    ROS_INFO_STREAM_THROTTLE(1.0, "[ FaultInjector ] t=" << std::fixed << std::setprecision(3)
                             << relative_sec << " scan=" << scan_index_
                             << " points=" << corrupted_points
                             << " scale=" << model_.rangeScale(relative_sec));
    output_pub_.publish(output);
    ++scan_index_;
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  fi::Model model_;
  ros::Subscriber input_sub_;
  ros::Publisher output_pub_;
  ros::Publisher active_pub_;
  ros::Publisher metadata_pub_;
  std::string input_topic_;
  std::string output_topic_;
  ros::Time epoch_stamp_;
  bool epoch_initialized_ = false;
  bool active_ = true; // Force the initial latched false publication.
  std::uint64_t scan_index_ = 0;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pointcloud_fault_injector");
  try
  {
    PointCloudFaultInjector injector;
    ros::spin();
  }
  catch (const std::exception &error)
  {
    ROS_FATAL_STREAM("[ FaultInjector ] " << error.what());
    return 2;
  }
  return 0;
}
