#ifndef FAST_LIVO_SHADOW_GENERATION_SELECTOR_H
#define FAST_LIVO_SHADOW_GENERATION_SELECTOR_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace fast_livo
{
namespace shadow_mux
{

enum class State
{
  INCUMBENT,
  PROBATION,
  CANDIDATE,
  ABORTED
};

struct Config
{
  double probation_duration_sec = 20.0;
  double timestamp_tolerance_sec = 0.05;
  double maximum_boundary_position_m = 0.50;
  double maximum_boundary_rotation_deg = 2.0;
  double maximum_position_growth_m = 0.50;
  double maximum_rotation_growth_deg = 5.0;
  std::size_t minimum_matched_samples = 50;
  double minimum_post_catchup_duration_sec = 2.0;
  std::size_t minimum_post_catchup_samples = 20;
  double maximum_position_variance_ratio = 3.0;
  double maximum_rotation_variance_ratio = 2.2;
  std::size_t minimum_consecutive_uncertainty_exceedance_samples = 1;
  bool align_candidate_at_boundary = false;
  // Default off. Hold a fixed translation throughout probation, then release
  // it smoothly only after the candidate is selected for public output.
  bool correction_aware_position_bridge_enabled = false;
  double position_bridge_duration_sec = 8.0;
  double maximum_added_bridge_speed_mps = 0.1;
  double maximum_handover_residual_position_m = 0.1;

  void validate() const
  {
    if (probation_duration_sec <= 0.0 || timestamp_tolerance_sec < 0.0 ||
        maximum_boundary_position_m <= 0.0 ||
        maximum_boundary_rotation_deg <= 0.0 ||
        maximum_position_growth_m <= 0.0 ||
        maximum_rotation_growth_deg <= 0.0 || minimum_matched_samples == 0)
      throw std::invalid_argument("invalid shadow-generation selector configuration");
    if (minimum_post_catchup_duration_sec < 0.0 ||
        minimum_post_catchup_samples == 0)
      throw std::invalid_argument("invalid post-catchup validation configuration");
    if (maximum_position_variance_ratio <= 0.0 ||
        maximum_rotation_variance_ratio <= 0.0 ||
        minimum_consecutive_uncertainty_exceedance_samples == 0)
      throw std::invalid_argument("invalid uncertainty-ratio configuration");
    if (correction_aware_position_bridge_enabled &&
        (align_candidate_at_boundary || position_bridge_duration_sec <= 0.0 ||
         maximum_added_bridge_speed_mps <= 0.0 ||
         maximum_handover_residual_position_m <= 0.0))
      throw std::invalid_argument("invalid correction-aware position bridge configuration");
  }
};

struct Pose
{
  double time = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double qw = 1.0;
  std::string frame_id;
  std::string child_frame_id;
  double position_variance = 1.0;
  double rotation_variance = 1.0;
};

struct Evaluation
{
  State state = State::INCUMBENT;
  std::string reason = "incumbent_active";
  bool candidate_rejected = false;
  std::string candidate_rejection_reason;
  std::size_t matched_samples = 0;
  double boundary_position_m = 0.0;
  double boundary_rotation_deg = 0.0;
  double raw_boundary_position_m = 0.0;
  double raw_boundary_rotation_deg = 0.0;
  bool boundary_alignment_applied = false;
  bool position_bridge_applied = false;
  double position_bridge_initial_offset_m = 0.0;
  double position_bridge_peak_added_speed_mps = 0.0;
  double position_bridge_release_start_sec =
      std::numeric_limits<double>::quiet_NaN();
  double handover_residual_position_m =
      std::numeric_limits<double>::quiet_NaN();
  double maximum_position_growth_m = 0.0;
  double maximum_rotation_growth_deg = 0.0;
  double probation_elapsed_sec = 0.0;
  std::size_t post_catchup_samples = 0;
  double post_catchup_elapsed_sec = 0.0;
  double maximum_position_variance_ratio = 0.0;
  double maximum_rotation_variance_ratio = 0.0;
  double second_maximum_position_variance_ratio = 0.0;
  double second_maximum_rotation_variance_ratio = 0.0;
  double mean_position_variance_ratio = 0.0;
  double mean_rotation_variance_ratio = 0.0;
  std::size_t uncertainty_samples = 0;
  std::size_t required_consecutive_uncertainty_exceedance_samples = 1;
  std::size_t position_uncertainty_exceedance_samples = 0;
  std::size_t rotation_uncertainty_exceedance_samples = 0;
  std::size_t current_consecutive_position_uncertainty_exceedances = 0;
  std::size_t current_consecutive_rotation_uncertainty_exceedances = 0;
  std::size_t maximum_consecutive_position_uncertainty_exceedances = 0;
  std::size_t maximum_consecutive_rotation_uncertainty_exceedances = 0;
  double position_variance_ratio_time_sec = 0.0;
  double candidate_position_variance_at_maximum_ratio = 0.0;
  double incumbent_position_variance_at_maximum_ratio = 0.0;
  double rotation_variance_ratio_time_sec = 0.0;
  double candidate_rotation_variance_at_maximum_ratio = 0.0;
  double incumbent_rotation_variance_at_maximum_ratio = 0.0;
  double candidate_commit_sensor_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  double candidate_commit_latest_incumbent_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  double candidate_commit_lag_sec =
      std::numeric_limits<double>::quiet_NaN();
  std::size_t precommit_candidate_samples_ignored = 0;
  double last_ignored_precommit_candidate_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  double first_evaluated_candidate_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  double first_evaluated_incumbent_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  double first_evaluated_match_dt_sec =
      std::numeric_limits<double>::quiet_NaN();
  double first_evaluated_candidate_minus_commit_sec =
      std::numeric_limits<double>::quiet_NaN();
  double maximum_absolute_match_dt_sec = 0.0;
};

class Selector
{
public:
  explicit Selector(const Config &config) : config_(config)
  {
    config_.validate();
    evaluation_.required_consecutive_uncertainty_exceedance_samples =
        config_.minimum_consecutive_uncertainty_exceedance_samples;
  }

  void candidateCommitted(
      const double commit_sensor_time_sec =
          std::numeric_limits<double>::quiet_NaN(),
      const double latest_published_incumbent_time =
          std::numeric_limits<double>::quiet_NaN())
  {
    if (evaluation_.state == State::INCUMBENT)
    {
      evaluation_ = Evaluation();
      evaluation_.required_consecutive_uncertainty_exceedance_samples =
          config_.minimum_consecutive_uncertainty_exceedance_samples;
      evaluation_.state = State::PROBATION;
      evaluation_.reason = "waiting_for_matched_candidate_output";
      evaluation_.candidate_commit_sensor_time_sec = commit_sensor_time_sec;
      evaluation_.candidate_commit_latest_incumbent_time_sec =
          latest_published_incumbent_time;
      if (std::isfinite(commit_sensor_time_sec) &&
          std::isfinite(latest_published_incumbent_time))
        evaluation_.candidate_commit_lag_sec =
            latest_published_incumbent_time - commit_sensor_time_sec;
      first_time_ = -1.0;
      post_catchup_first_time_ = -1.0;
      ever_caught_up_ = false;
      boundary_alignment_initialized_ = false;
      position_bridge_initialized_ = false;
      position_bridge_release_start_time_ =
          std::numeric_limits<double>::quiet_NaN();
    }
  }

  void candidateRejected(const std::string &reason)
  {
    if (evaluation_.state == State::CANDIDATE) return;
    evaluation_.state = State::INCUMBENT;
    evaluation_.reason = "incumbent_retained_after_candidate_rejection";
    evaluation_.candidate_rejected = true;
    evaluation_.candidate_rejection_reason =
        reason.empty() ? "candidate_rejected" : reason;
  }

  const Evaluation &observe(const Pose &incumbent, const Pose &candidate,
                            const double latest_published_incumbent_time)
  {
    if (evaluation_.state != State::PROBATION) return evaluation_;
    if (!finite(incumbent) || !finite(candidate))
      return abort("non_finite_pose");
    // Candidate odometry and lifecycle status use independent ROS subscriber
    // queues. A commit notification can therefore overtake buffered replay
    // odometry. Those pre-commit poses describe the provisional replay history,
    // not the future generation that probation is meant to validate.
    if (std::isfinite(evaluation_.candidate_commit_sensor_time_sec) &&
        candidate.time + 1e-9 < evaluation_.candidate_commit_sensor_time_sec)
    {
      ++evaluation_.precommit_candidate_samples_ignored;
      evaluation_.last_ignored_precommit_candidate_time_sec = candidate.time;
      evaluation_.reason = "waiting_for_post_commit_candidate_output";
      return evaluation_;
    }
    if (std::abs(incumbent.time - candidate.time) > config_.timestamp_tolerance_sec)
      return evaluation_;
    if (incumbent.frame_id != candidate.frame_id ||
        incumbent.child_frame_id != candidate.child_frame_id)
      return abort("frame_id_mismatch");

    const double signed_match_dt = candidate.time - incumbent.time;
    if (!std::isfinite(evaluation_.first_evaluated_candidate_time_sec))
    {
      evaluation_.first_evaluated_candidate_time_sec = candidate.time;
      evaluation_.first_evaluated_incumbent_time_sec = incumbent.time;
      evaluation_.first_evaluated_match_dt_sec = signed_match_dt;
      if (std::isfinite(evaluation_.candidate_commit_sensor_time_sec))
        evaluation_.first_evaluated_candidate_minus_commit_sec =
            candidate.time - evaluation_.candidate_commit_sensor_time_sec;
    }
    evaluation_.maximum_absolute_match_dt_sec = std::max(
        evaluation_.maximum_absolute_match_dt_sec, std::abs(signed_match_dt));

    const double raw_position = positionDivergence(incumbent, candidate);
    const double raw_rotation = rotationDivergenceDeg(incumbent, candidate);
    if (first_time_ < 0.0 && config_.align_candidate_at_boundary)
      initializeBoundaryAlignment(incumbent, candidate);
    if (first_time_ < 0.0 && config_.correction_aware_position_bridge_enabled)
      initializePositionBridge(incumbent, candidate);
    const Pose comparison_candidate = alignedCandidatePose(candidate);
    const double position = positionDivergence(incumbent, comparison_candidate);
    const double rotation = rotationDivergenceDeg(incumbent, comparison_candidate);
    const double position_variance_ratio = varianceRatio(
        candidate.position_variance, incumbent.position_variance);
    const double rotation_variance_ratio = varianceRatio(
        candidate.rotation_variance, incumbent.rotation_variance);
    ++evaluation_.uncertainty_samples;
    evaluation_.mean_position_variance_ratio +=
        (position_variance_ratio - evaluation_.mean_position_variance_ratio) /
        static_cast<double>(evaluation_.uncertainty_samples);
    evaluation_.mean_rotation_variance_ratio +=
        (rotation_variance_ratio - evaluation_.mean_rotation_variance_ratio) /
        static_cast<double>(evaluation_.uncertainty_samples);
    if (position_variance_ratio > evaluation_.maximum_position_variance_ratio)
    {
      evaluation_.second_maximum_position_variance_ratio =
          evaluation_.maximum_position_variance_ratio;
      evaluation_.maximum_position_variance_ratio = position_variance_ratio;
      evaluation_.position_variance_ratio_time_sec = candidate.time;
      evaluation_.candidate_position_variance_at_maximum_ratio =
          candidate.position_variance;
      evaluation_.incumbent_position_variance_at_maximum_ratio =
          incumbent.position_variance;
    }
    else if (position_variance_ratio >
             evaluation_.second_maximum_position_variance_ratio)
    {
      evaluation_.second_maximum_position_variance_ratio =
          position_variance_ratio;
    }
    if (rotation_variance_ratio > evaluation_.maximum_rotation_variance_ratio)
    {
      evaluation_.second_maximum_rotation_variance_ratio =
          evaluation_.maximum_rotation_variance_ratio;
      evaluation_.maximum_rotation_variance_ratio = rotation_variance_ratio;
      evaluation_.rotation_variance_ratio_time_sec = candidate.time;
      evaluation_.candidate_rotation_variance_at_maximum_ratio =
          candidate.rotation_variance;
      evaluation_.incumbent_rotation_variance_at_maximum_ratio =
          incumbent.rotation_variance;
    }
    else if (rotation_variance_ratio >
             evaluation_.second_maximum_rotation_variance_ratio)
    {
      evaluation_.second_maximum_rotation_variance_ratio =
          rotation_variance_ratio;
    }
    updateUncertaintyExceedance(
        position_variance_ratio > config_.maximum_position_variance_ratio,
        evaluation_.position_uncertainty_exceedance_samples,
        evaluation_.current_consecutive_position_uncertainty_exceedances,
        evaluation_.maximum_consecutive_position_uncertainty_exceedances);
    updateUncertaintyExceedance(
        rotation_variance_ratio > config_.maximum_rotation_variance_ratio,
        evaluation_.rotation_uncertainty_exceedance_samples,
        evaluation_.current_consecutive_rotation_uncertainty_exceedances,
        evaluation_.maximum_consecutive_rotation_uncertainty_exceedances);
    if (evaluation_.current_consecutive_position_uncertainty_exceedances >=
        config_.minimum_consecutive_uncertainty_exceedance_samples)
      return abort("position_uncertainty_ratio_limit");
    if (evaluation_.current_consecutive_rotation_uncertainty_exceedances >=
        config_.minimum_consecutive_uncertainty_exceedance_samples)
      return abort("rotation_uncertainty_ratio_limit");
    if (first_time_ < 0.0)
    {
      first_time_ = candidate.time;
      evaluation_.raw_boundary_position_m = raw_position;
      evaluation_.raw_boundary_rotation_deg = raw_rotation;
      evaluation_.boundary_alignment_applied = boundary_alignment_initialized_;
      evaluation_.position_bridge_applied = position_bridge_initialized_;
      if (position_bridge_initialized_)
      {
        evaluation_.position_bridge_initial_offset_m = raw_position;
        evaluation_.position_bridge_peak_added_speed_mps =
            1.5 * raw_position / config_.position_bridge_duration_sec;
        if (evaluation_.position_bridge_peak_added_speed_mps >
            config_.maximum_added_bridge_speed_mps)
          return abort("bridge_added_speed_limit");
      }
      evaluation_.boundary_position_m = position;
      evaluation_.boundary_rotation_deg = rotation;
      if (position > config_.maximum_boundary_position_m)
        return abort("boundary_position_limit");
      if (rotation > config_.maximum_boundary_rotation_deg)
        return abort("boundary_rotation_limit");
    }

    ++evaluation_.matched_samples;
    evaluation_.probation_elapsed_sec = std::max(0.0, candidate.time - first_time_);
    evaluation_.maximum_position_growth_m = std::max(
        evaluation_.maximum_position_growth_m,
        std::max(0.0, config_.correction_aware_position_bridge_enabled ?
            raw_position - evaluation_.raw_boundary_position_m :
            position - evaluation_.boundary_position_m));
    evaluation_.maximum_rotation_growth_deg = std::max(
        evaluation_.maximum_rotation_growth_deg,
        std::max(0.0, rotation - evaluation_.boundary_rotation_deg));
    if (evaluation_.maximum_position_growth_m > config_.maximum_position_growth_m)
      return abort("position_growth_limit");
    if (evaluation_.maximum_rotation_growth_deg > config_.maximum_rotation_growth_deg)
      return abort("rotation_growth_limit");

    const bool base_probation_complete =
        evaluation_.probation_elapsed_sec >= config_.probation_duration_sec &&
        evaluation_.matched_samples >= config_.minimum_matched_samples;
    if (base_probation_complete)
    {
      const bool caught_up = candidate.time + config_.timestamp_tolerance_sec >=
          latest_published_incumbent_time;
      if (!caught_up)
      {
        if (post_catchup_first_time_ >= 0.0)
        {
          post_catchup_first_time_ = -1.0;
          evaluation_.post_catchup_samples = 0;
          evaluation_.post_catchup_elapsed_sec = 0.0;
          evaluation_.reason = "candidate_lost_catchup";
        }
        else
        {
          evaluation_.reason = "candidate_not_caught_up";
        }
      }
      else
      {
        ever_caught_up_ = true;
        if (post_catchup_first_time_ < 0.0)
          post_catchup_first_time_ = candidate.time;
        ++evaluation_.post_catchup_samples;
        evaluation_.post_catchup_elapsed_sec =
            std::max(0.0, candidate.time - post_catchup_first_time_);
        if (evaluation_.post_catchup_elapsed_sec >=
                config_.minimum_post_catchup_duration_sec &&
            evaluation_.post_catchup_samples >=
                config_.minimum_post_catchup_samples)
        {
          if (config_.correction_aware_position_bridge_enabled)
          {
            evaluation_.handover_residual_position_m = position;
            if (position > config_.maximum_handover_residual_position_m)
              return abort("handover_residual_position_limit");
          }
          evaluation_.state = State::CANDIDATE;
          evaluation_.reason = "healthy_tail_approved";
        }
        else
        {
          evaluation_.reason = "waiting_for_post_catchup_tail";
        }
      }
    }
    else
    {
      evaluation_.reason = "probation_in_progress";
    }
    return evaluation_;
  }

  const Evaluation &finalize()
  {
    if (evaluation_.state != State::PROBATION) return evaluation_;
    return abort(ever_caught_up_ ? "terminal_only_handover"
                                 : "candidate_not_caught_up_at_end");
  }

  const Evaluation &evaluation() const { return evaluation_; }
  double timestampToleranceSec() const { return config_.timestamp_tolerance_sec; }

  void resetForNextCandidate()
  {
    if (evaluation_.state != State::ABORTED &&
        !(evaluation_.state == State::INCUMBENT &&
          evaluation_.candidate_rejected))
      throw std::logic_error("only a rejected or aborted candidate can be re-armed");
    *this = Selector(config_);
  }

  Pose alignedCandidatePose(const Pose &candidate) const
  {
    if (position_bridge_initialized_)
    {
      Pose bridged = candidate;
      const double retained = positionBridgeRetainedFraction(candidate.time);
      bridged.x += retained * position_bridge_x_;
      bridged.y += retained * position_bridge_y_;
      bridged.z += retained * position_bridge_z_;
      return bridged;
    }
    if (!boundary_alignment_initialized_) return candidate;
    Pose aligned = candidate;
    rotate(alignment_qx_, alignment_qy_, alignment_qz_, alignment_qw_,
           candidate.x, candidate.y, candidate.z,
           aligned.x, aligned.y, aligned.z);
    aligned.x += alignment_tx_;
    aligned.y += alignment_ty_;
    aligned.z += alignment_tz_;
    multiplyQuaternion(
        alignment_qx_, alignment_qy_, alignment_qz_, alignment_qw_,
        candidate.qx, candidate.qy, candidate.qz, candidate.qw,
        aligned.qx, aligned.qy, aligned.qz, aligned.qw);
    normalizeQuaternion(aligned.qx, aligned.qy, aligned.qz, aligned.qw);
    return aligned;
  }

  bool boundaryAlignmentInitialized() const
  {
    return boundary_alignment_initialized_ || position_bridge_initialized_;
  }

  bool positionBridgeInitialized() const { return position_bridge_initialized_; }
  bool positionBridgeEnabled() const
  {
    return config_.correction_aware_position_bridge_enabled;
  }
  double positionBridgeDurationSec() const
  {
    return config_.position_bridge_duration_sec;
  }
  double maximumHandoverResidualPositionM() const
  {
    return config_.maximum_handover_residual_position_m;
  }

  void startPositionBridgeRelease(const double first_public_candidate_time)
  {
    if (!config_.correction_aware_position_bridge_enabled ||
        !position_bridge_initialized_ || evaluation_.state != State::CANDIDATE ||
        std::isfinite(position_bridge_release_start_time_) ||
        !std::isfinite(first_public_candidate_time))
      throw std::logic_error("position bridge release requires first public candidate pose");
    position_bridge_release_start_time_ = first_public_candidate_time;
    evaluation_.position_bridge_release_start_sec = first_public_candidate_time;
  }

  void abortBeforePublicCommit(const std::string &reason)
  {
    if (evaluation_.state != State::CANDIDATE ||
        std::isfinite(position_bridge_release_start_time_))
      throw std::logic_error("cannot abort an already public candidate");
    abort(reason);
  }

  double positionBridgeRetainedFraction(const double time) const
  {
    if (!position_bridge_initialized_) return 0.0;
    if (!std::isfinite(position_bridge_release_start_time_) ||
        time <= position_bridge_release_start_time_)
      return 1.0;
    const double u = std::max(0.0, std::min(1.0,
        (time - position_bridge_release_start_time_) /
        config_.position_bridge_duration_sec));
    return 1.0 - (3.0 * u * u - 2.0 * u * u * u);
  }

private:
  static void normalizeQuaternion(double &x, double &y, double &z, double &w)
  {
    const double norm = std::sqrt(x * x + y * y + z * z + w * w);
    if (norm <= 1e-12) return;
    x /= norm;
    y /= norm;
    z /= norm;
    w /= norm;
  }

  static void multiplyQuaternion(
      const double ax, const double ay, const double az, const double aw,
      const double bx, const double by, const double bz, const double bw,
      double &x, double &y, double &z, double &w)
  {
    x = aw * bx + ax * bw + ay * bz - az * by;
    y = aw * by - ax * bz + ay * bw + az * bx;
    z = aw * bz + ax * by - ay * bx + az * bw;
    w = aw * bw - ax * bx - ay * by - az * bz;
  }

  static void rotate(
      double qx, double qy, double qz, double qw,
      const double x, const double y, const double z,
      double &out_x, double &out_y, double &out_z)
  {
    normalizeQuaternion(qx, qy, qz, qw);
    const double tx = 2.0 * (qy * z - qz * y);
    const double ty = 2.0 * (qz * x - qx * z);
    const double tz = 2.0 * (qx * y - qy * x);
    out_x = x + qw * tx + (qy * tz - qz * ty);
    out_y = y + qw * ty + (qz * tx - qx * tz);
    out_z = z + qw * tz + (qx * ty - qy * tx);
  }

  void initializeBoundaryAlignment(const Pose &incumbent,
                                   const Pose &candidate)
  {
    double ix = incumbent.qx, iy = incumbent.qy, iz = incumbent.qz,
           iw = incumbent.qw;
    double cx = candidate.qx, cy = candidate.qy, cz = candidate.qz,
           cw = candidate.qw;
    normalizeQuaternion(ix, iy, iz, iw);
    normalizeQuaternion(cx, cy, cz, cw);
    multiplyQuaternion(ix, iy, iz, iw, -cx, -cy, -cz, cw,
                       alignment_qx_, alignment_qy_, alignment_qz_,
                       alignment_qw_);
    normalizeQuaternion(alignment_qx_, alignment_qy_, alignment_qz_,
                        alignment_qw_);
    double rotated_x = 0.0, rotated_y = 0.0, rotated_z = 0.0;
    rotate(alignment_qx_, alignment_qy_, alignment_qz_, alignment_qw_,
           candidate.x, candidate.y, candidate.z,
           rotated_x, rotated_y, rotated_z);
    alignment_tx_ = incumbent.x - rotated_x;
    alignment_ty_ = incumbent.y - rotated_y;
    alignment_tz_ = incumbent.z - rotated_z;
    boundary_alignment_initialized_ = true;
  }

  void initializePositionBridge(const Pose &incumbent, const Pose &candidate)
  {
    position_bridge_x_ = incumbent.x - candidate.x;
    position_bridge_y_ = incumbent.y - candidate.y;
    position_bridge_z_ = incumbent.z - candidate.z;
    position_bridge_initialized_ = true;
  }

  static void updateUncertaintyExceedance(
      const bool exceeds, std::size_t &total, std::size_t &current,
      std::size_t &maximum)
  {
    if (exceeds)
    {
      ++total;
      ++current;
      maximum = std::max(maximum, current);
    }
    else
    {
      current = 0;
    }
  }

  static bool finite(const Pose &pose)
  {
    return std::isfinite(pose.time) && std::isfinite(pose.x) &&
        std::isfinite(pose.y) && std::isfinite(pose.z) &&
        std::isfinite(pose.qx) && std::isfinite(pose.qy) &&
        std::isfinite(pose.qz) && std::isfinite(pose.qw);
  }

  static double positionDivergence(const Pose &a, const Pose &b)
  {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  static double varianceRatio(const double candidate, const double incumbent)
  {
    if (!std::isfinite(candidate) || !std::isfinite(incumbent) ||
        candidate < 0.0 || incumbent < 0.0)
      return std::numeric_limits<double>::infinity();
    return candidate / std::max(incumbent, 1e-12);
  }

  static double rotationDivergenceDeg(const Pose &a, const Pose &b)
  {
    const double an = std::sqrt(a.qx * a.qx + a.qy * a.qy +
                                a.qz * a.qz + a.qw * a.qw);
    const double bn = std::sqrt(b.qx * b.qx + b.qy * b.qy +
                                b.qz * b.qz + b.qw * b.qw);
    if (an <= 1e-12 || bn <= 1e-12)
      return 180.0;
    double dot = (a.qx * b.qx + a.qy * b.qy + a.qz * b.qz +
                  a.qw * b.qw) / (an * bn);
    dot = std::max(-1.0, std::min(1.0, std::abs(dot)));
    return 2.0 * std::acos(dot) * 180.0 / 3.14159265358979323846;
  }

  const Evaluation &abort(const std::string &reason)
  {
    evaluation_.state = State::ABORTED;
    evaluation_.reason = reason;
    return evaluation_;
  }

  Config config_;
  Evaluation evaluation_;
  double first_time_ = -1.0;
  double post_catchup_first_time_ = -1.0;
  bool ever_caught_up_ = false;
  bool boundary_alignment_initialized_ = false;
  bool position_bridge_initialized_ = false;
  double position_bridge_x_ = 0.0;
  double position_bridge_y_ = 0.0;
  double position_bridge_z_ = 0.0;
  double position_bridge_release_start_time_ =
      std::numeric_limits<double>::quiet_NaN();
  double alignment_qx_ = 0.0;
  double alignment_qy_ = 0.0;
  double alignment_qz_ = 0.0;
  double alignment_qw_ = 1.0;
  double alignment_tx_ = 0.0;
  double alignment_ty_ = 0.0;
  double alignment_tz_ = 0.0;
};

inline const char *stateName(const State state)
{
  switch (state)
  {
    case State::INCUMBENT: return "incumbent";
    case State::PROBATION: return "probation";
    case State::CANDIDATE: return "candidate";
    case State::ABORTED: return "aborted";
  }
  return "unknown";
}

} // namespace shadow_mux
} // namespace fast_livo

#endif
