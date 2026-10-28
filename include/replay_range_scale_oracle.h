#ifndef REPLAY_RANGE_SCALE_ORACLE_H_
#define REPLAY_RANGE_SCALE_ORACLE_H_

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fast_livo
{
namespace replay_oracle
{

// PointXYZINormal's input normal fields are unused by the LiDAR estimator in
// this project. When the diagnostic oracle is enabled, they carry the exact
// scan-level inverse scale and source-scan time through LIVO aggregation and
// disk-backed replay. The marker prevents ordinary point normals from being
// interpreted as oracle metadata.
constexpr float kRangeScaleProvenanceMarker = -32016.0f;

inline bool hasRangeScaleProvenance(const float marker)
{
  return std::isfinite(marker) &&
      std::abs(marker - kRangeScaleProvenanceMarker) < 0.25f;
}

inline bool validEncodedInverseScale(const float inverse_scale)
{
  return std::isfinite(inverse_scale) && inverse_scale > 0.0f &&
      inverse_scale <= 4.0f;
}

// Diagnostic-only inverse model for establishing the upper bound of replay
// recovery. It deliberately consumes the injected fault schedule and must not
// be enabled in held-out or final paper experiments.
struct RangeScaleConfig
{
  bool enabled = false;
  double start_sec = 0.0;
  double end_sec = 0.0;
  double scale_start = 1.0;
  double scale_end = 1.0;

  void validate() const
  {
    if (!enabled) return;
    if (!std::isfinite(start_sec) || !std::isfinite(end_sec) ||
        end_sec <= start_sec)
      throw std::invalid_argument(
          "oracle range-scale interval must satisfy start < end");
    if (!std::isfinite(scale_start) || !std::isfinite(scale_end) ||
        scale_start <= 0.0 || scale_end <= 0.0)
      throw std::invalid_argument("oracle range scales must be positive");
  }

  bool active(const double relative_sec) const
  {
    return enabled && relative_sec >= start_sec && relative_sec <= end_sec;
  }

  double scale(const double relative_sec) const
  {
    const double phase = std::max(
        0.0, std::min(1.0,
                      (relative_sec - start_sec) / (end_sec - start_sec)));
    return scale_start + phase * (scale_end - scale_start);
  }

  double inverseScale(const double relative_sec) const
  {
    return active(relative_sec) ? 1.0 / scale(relative_sec) : 1.0;
  }
};

struct RawScanCorrection
{
  bool active = false;
  double inverse_scale = 1.0;
};

// Only the private replay diagnostic uses this schedule. Raw source messages
// remain immutable and no live-cloud provenance fields are changed. Archived
// oracle metadata remains the legacy fallback when the new mode is disabled.
inline RawScanCorrection rawScanCorrection(
    const RangeScaleConfig &profile, const double scan_relative_sec,
    const bool archived_active, const double archived_inverse_scale)
{
  if (!profile.enabled) return {archived_active, archived_inverse_scale};
  if (!std::isfinite(scan_relative_sec))
    throw std::invalid_argument("oracle raw scan time is not finite");
  const double inverse = profile.inverseScale(scan_relative_sec);
  return {profile.active(scan_relative_sec) && inverse != 1.0, inverse};
}

} // namespace replay_oracle
} // namespace fast_livo

#endif // REPLAY_RANGE_SCALE_ORACLE_H_
