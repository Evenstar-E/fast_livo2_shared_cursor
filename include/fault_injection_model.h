#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fast_livo
{
namespace fault_injection
{

enum class LidarFaultMode
{
  NONE,
  RANGE_SCALE_DRIFT,
  RIGID_GHOST,
  MIXED
};

inline LidarFaultMode parseMode(const std::string &name)
{
  if (name == "none") return LidarFaultMode::NONE;
  if (name == "range_scale_drift") return LidarFaultMode::RANGE_SCALE_DRIFT;
  if (name == "rigid_ghost") return LidarFaultMode::RIGID_GHOST;
  if (name == "mixed") return LidarFaultMode::MIXED;
  throw std::invalid_argument("unknown LiDAR fault mode: " + name);
}

inline const char *modeName(LidarFaultMode mode)
{
  switch (mode)
  {
    case LidarFaultMode::NONE: return "none";
    case LidarFaultMode::RANGE_SCALE_DRIFT: return "range_scale_drift";
    case LidarFaultMode::RIGID_GHOST: return "rigid_ghost";
    case LidarFaultMode::MIXED: return "mixed";
  }
  return "none";
}

struct Config
{
  double start_sec = 30.0;
  double end_sec = 36.0;
  double range_scale_start = 1.0;
  double range_scale_end = 1.04;
  double ghost_fraction = 0.35;
  double ghost_yaw_deg = 2.0;
  double ghost_translation_x = 0.20;
  double ghost_translation_y = 0.0;
  double ghost_translation_z = 0.0;
  double point_time_scale = 1.0;
  std::uint64_t seed = 20260909ULL;
  LidarFaultMode mode = LidarFaultMode::MIXED;
};

class Model
{
public:
  explicit Model(Config config) : config_(config)
  {
    if (config_.start_sec < 0.0 || config_.end_sec <= config_.start_sec)
      throw std::invalid_argument("fault interval must satisfy 0 <= start < end");
    if (config_.range_scale_start <= 0.0 || config_.range_scale_end <= 0.0)
      throw std::invalid_argument("range scales must be positive");
    if (config_.ghost_fraction < 0.0 || config_.ghost_fraction > 1.0)
      throw std::invalid_argument("ghost_fraction must be in [0, 1]");
    if (config_.point_time_scale <= 0.0)
      throw std::invalid_argument("point_time_scale must be positive");
  }

  const Config &config() const { return config_; }

  bool active(double relative_sec) const
  {
    return config_.mode != LidarFaultMode::NONE &&
           relative_sec >= config_.start_sec && relative_sec <= config_.end_sec;
  }

  double phase(double relative_sec) const
  {
    return std::max(0.0, std::min(1.0,
      (relative_sec - config_.start_sec) / (config_.end_sec - config_.start_sec)));
  }

  double rangeScale(double relative_sec) const
  {
    const double alpha = phase(relative_sec);
    return config_.range_scale_start +
           alpha * (config_.range_scale_end - config_.range_scale_start);
  }

  bool appliesRangeScale() const
  {
    return config_.mode == LidarFaultMode::RANGE_SCALE_DRIFT ||
           config_.mode == LidarFaultMode::MIXED;
  }

  bool appliesRigidGhost() const
  {
    return config_.mode == LidarFaultMode::RIGID_GHOST ||
           config_.mode == LidarFaultMode::MIXED;
  }

  bool ghostSelected(std::size_t point_index, std::uint64_t scan_index) const
  {
    std::uint64_t key = config_.seed;
    key ^= scan_index * 0x9e3779b97f4a7c15ULL;
    key ^= static_cast<std::uint64_t>(point_index) * 0xbf58476d1ce4e5b9ULL;
    const std::uint64_t value = splitMix64(key);
    const double unit = static_cast<double>(value >> 11) * (1.0 / 9007199254740992.0);
    return unit < config_.ghost_fraction;
  }

  void corruptPoint(double relative_sec, std::size_t point_index,
                    std::uint64_t scan_index, double &x, double &y, double &z) const
  {
    if (appliesRangeScale())
    {
      const double scale = rangeScale(relative_sec);
      x *= scale;
      y *= scale;
      z *= scale;
    }

    if (appliesRigidGhost() && ghostSelected(point_index, scan_index))
    {
      const double yaw = config_.ghost_yaw_deg * 3.14159265358979323846 / 180.0;
      const double c = std::cos(yaw);
      const double s = std::sin(yaw);
      const double rotated_x = c * x - s * y;
      const double rotated_y = s * x + c * y;
      x = rotated_x + config_.ghost_translation_x;
      y = rotated_y + config_.ghost_translation_y;
      z += config_.ghost_translation_z;
    }
  }

private:
  static std::uint64_t splitMix64(std::uint64_t value)
  {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  Config config_;
};

} // namespace fault_injection
} // namespace fast_livo
