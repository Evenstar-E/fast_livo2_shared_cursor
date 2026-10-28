#ifndef FAST_LIVO_INCREMENTAL_PUBLIC_GUARD_H
#define FAST_LIVO_INCREMENTAL_PUBLIC_GUARD_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace fast_livo {
namespace incremental_public_guard {

struct Config {
  bool enabled = false;
  double minimum_onset_confidence = 0.60;
  double horizon_sec = 8.0;
  double lidar_information_weight = 0.10;
  double terminal_range_scale = 1.04;
  std::size_t minimum_probe_matches = 80;
  std::size_t consecutive_probe_passes = 3;
  double minimum_probe_relative_improvement = 0.20;
  double minimum_probe_absolute_improvement_m = 0.004;

  void validate() const {
    if (!std::isfinite(minimum_onset_confidence) ||
        minimum_onset_confidence < 0.0 || minimum_onset_confidence > 1.0 ||
        !std::isfinite(horizon_sec) || horizon_sec <= 0.0 ||
        horizon_sec > 30.0 ||
        !std::isfinite(lidar_information_weight) ||
        lidar_information_weight <= 0.0 || lidar_information_weight > 1.0 ||
        !std::isfinite(terminal_range_scale) || terminal_range_scale <= 1.0 ||
        terminal_range_scale > 1.10 || minimum_probe_matches == 0 ||
        consecutive_probe_passes == 0 || consecutive_probe_passes > 10 ||
        !std::isfinite(minimum_probe_relative_improvement) ||
        minimum_probe_relative_improvement < 0.0 ||
        minimum_probe_relative_improvement >= 1.0 ||
        !std::isfinite(minimum_probe_absolute_improvement_m) ||
        minimum_probe_absolute_improvement_m < 0.0)
      throw std::invalid_argument("invalid incremental public guard configuration");
  }
};

// The online scale hypothesis must beat the identity control on the same
// current scan and same pre-update map. Onset confidence alone is not fault
// probability and must never authorize a public-state intervention.
struct ProbeResult {
  std::size_t identity_matches = 0;
  std::size_t corrected_matches = 0;
  double identity_mean_abs_residual = std::numeric_limits<double>::infinity();
  double corrected_mean_abs_residual = std::numeric_limits<double>::infinity();
  bool valid = false;
};

class EvidenceTracker {
 public:
  explicit EvidenceTracker(const Config &config = Config()) : config_(config) {
    config_.validate();
  }
  void configure(const Config &config) {
    config.validate();
    config_ = config;
    reset();
  }
  void reset() { passes_ = 0; episode_id_ = 0; }
  bool observe(std::size_t episode_id, const ProbeResult &probe) {
    if (!config_.enabled || !probe.valid ||
        probe.identity_matches < config_.minimum_probe_matches ||
        probe.corrected_matches < config_.minimum_probe_matches ||
        !std::isfinite(probe.identity_mean_abs_residual) ||
        !std::isfinite(probe.corrected_mean_abs_residual) ||
        probe.identity_mean_abs_residual <= 0.0 ||
        probe.corrected_mean_abs_residual < 0.0) {
      passes_ = 0;
      return false;
    }
    if (episode_id != episode_id_) {
      episode_id_ = episode_id;
      passes_ = 0;
    }
    const double absolute = probe.identity_mean_abs_residual -
                            probe.corrected_mean_abs_residual;
    const double relative = absolute / probe.identity_mean_abs_residual;
    if (absolute < config_.minimum_probe_absolute_improvement_m ||
        relative < config_.minimum_probe_relative_improvement) {
      passes_ = 0;
      return false;
    }
    if (passes_ < config_.consecutive_probe_passes) ++passes_;
    return passes_ >= config_.consecutive_probe_passes;
  }
  std::size_t passes() const { return passes_; }

 private:
  Config config_;
  std::size_t passes_ = 0;
  std::size_t episode_id_ = 0;
};

// A bounded, online-only provisional measurement policy.  It never changes
// previously published samples or grants permission for a candidate map
// commit; the normal replay and healthy-tail validator retain that authority.
class Guard {
 public:
  explicit Guard(const Config &config = Config()) : config_(config) {
    config_.validate();
  }

  void configure(const Config &config) {
    config.validate();
    config_ = config;
    start_sec_ = -1.0;
    end_sec_ = -1.0;
    last_episode_id_ = 0;
    ever_armed_ = false;
    expiry_reported_ = true;
  }

  bool arm(double detection_sec, double onset_confidence,
           std::size_t episode_id) {
    if (!config_.enabled || !std::isfinite(detection_sec) ||
        !std::isfinite(onset_confidence) ||
        onset_confidence < config_.minimum_onset_confidence ||
        onset_confidence > 1.0 ||
        (ever_armed_ && episode_id == last_episode_id_) ||
        active(detection_sec))
      return false;
    start_sec_ = detection_sec;
    end_sec_ = detection_sec + config_.horizon_sec;
    last_episode_id_ = episode_id;
    ever_armed_ = true;
    expiry_reported_ = false;
    return true;
  }

  bool active(double sensor_relative_sec) const {
    return config_.enabled && std::isfinite(sensor_relative_sec) &&
           sensor_relative_sec > start_sec_ &&
           sensor_relative_sec < end_sec_;
  }

  bool consumeExpiry(double sensor_relative_sec) {
    if (expiry_reported_ || !std::isfinite(sensor_relative_sec) ||
        sensor_relative_sec < end_sec_)
      return false;
    expiry_reported_ = true;
    return true;
  }

  double startSec() const { return start_sec_; }
  double endSec() const { return end_sec_; }
  double lidarInformationWeight() const {
    return config_.lidar_information_weight;
  }

 private:
  Config config_;
  double start_sec_ = -1.0;
  double end_sec_ = -1.0;
  std::size_t last_episode_id_ = 0;
  bool ever_armed_ = false;
  bool expiry_reported_ = true;
};

}  // namespace incremental_public_guard
}  // namespace fast_livo

#endif
