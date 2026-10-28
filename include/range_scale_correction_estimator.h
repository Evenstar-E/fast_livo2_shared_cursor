#ifndef RANGE_SCALE_CORRECTION_ESTIMATOR_H_
#define RANGE_SCALE_CORRECTION_ESTIMATOR_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fast_livo
{
namespace correction_estimation
{

// This component selects a correction only from causal candidate-branch health
// evidence.  It deliberately has no clean-map, injected-fault, or oracle
// interval input.  A rejected estimate is the expected fail-closed outcome.
struct Config
{
  bool enabled = false;
  double minimum_range_scale = 0.90;
  double maximum_range_scale = 1.10;
  double identity_range_scale = 1.0;
  double identity_tolerance = 1e-6;
  std::size_t minimum_hypotheses = 3;
  std::size_t minimum_samples = 20;
  double minimum_onset_confidence = 0.60;
  double minimum_relative_improvement = 0.10;
  double minimum_runner_up_margin = 0.04;
  double near_optimal_score_delta = 0.04;
  double maximum_confidence_interval_width = 0.04;
  double minimum_combined_confidence = 0.60;
  double residual_weight = 1.0;
  double match_weight = 1.0;
  double innovation_weight = 1.0;
  double match_error_floor = 0.02;
  double ratio_cap = 10.0;

  void validate() const
  {
    if (!enabled) return;
    if (!std::isfinite(minimum_range_scale) ||
        !std::isfinite(maximum_range_scale) ||
        !std::isfinite(identity_range_scale) ||
        minimum_range_scale <= 0.0 ||
        maximum_range_scale <= minimum_range_scale ||
        identity_range_scale < minimum_range_scale ||
        identity_range_scale > maximum_range_scale ||
        identity_tolerance < 0.0)
      throw std::invalid_argument("invalid range-scale search interval");
    if (minimum_hypotheses < 2 || minimum_samples == 0)
      throw std::invalid_argument("insufficient estimator evidence requirements");
    if (minimum_onset_confidence < 0.0 || minimum_onset_confidence > 1.0 ||
        minimum_relative_improvement < 0.0 ||
        minimum_relative_improvement >= 1.0 ||
        minimum_runner_up_margin < 0.0 ||
        near_optimal_score_delta < 0.0 ||
        maximum_confidence_interval_width < 0.0 ||
        minimum_combined_confidence < 0.0 ||
        minimum_combined_confidence > 1.0)
      throw std::invalid_argument("invalid estimator admission thresholds");
    if (residual_weight < 0.0 || match_weight < 0.0 ||
        innovation_weight < 0.0 ||
        residual_weight + match_weight + innovation_weight <= 0.0 ||
        match_error_floor <= 0.0 || ratio_cap <= 1.0)
      throw std::invalid_argument("invalid estimator metric normalization");
  }
};

struct OnsetInterval
{
  double lower_bound_sec = -1.0;
  double point_estimate_sec = -1.0;
  double upper_bound_sec = -1.0;
  double confidence = 0.0;
  double detection_sec = -1.0;

  bool valid() const
  {
    return std::isfinite(lower_bound_sec) &&
        std::isfinite(point_estimate_sec) &&
        std::isfinite(upper_bound_sec) && std::isfinite(confidence) &&
        std::isfinite(detection_sec) && lower_bound_sec >= 0.0 &&
        lower_bound_sec <= point_estimate_sec &&
        point_estimate_sec <= upper_bound_sec &&
        upper_bound_sec <= detection_sec && confidence >= 0.0 &&
        confidence <= 1.0;
  }
};

struct HypothesisEvidence
{
  // Terminal multiplicative corruption scale. The inverse is applied during
  // reconstruction. A scale of 1.0 is the mandatory no-correction control.
  double range_scale = 1.0;
  double residual_median = 0.0;
  double match_ratio_median = 0.0;
  double innovation_median = 0.0;
  std::size_t sample_count = 0;
  bool healthy_tail_pass = false;
  bool valid = true;
};

// Robustly reduces a branch's residual, match-quality, and innovation history
// to the evidence record consumed by the selector.
class HypothesisHistory
{
public:
  void observe(const double residual, const double match_ratio,
               const double innovation)
  {
    if (!std::isfinite(residual) || !std::isfinite(match_ratio) ||
        !std::isfinite(innovation) || residual < 0.0 || innovation < 0.0 ||
        match_ratio < 0.0 || match_ratio > 1.0)
      return;
    residuals_.push_back(residual);
    match_ratios_.push_back(match_ratio);
    innovations_.push_back(innovation);
  }

  std::size_t size() const { return residuals_.size(); }

  HypothesisEvidence summarize(const double range_scale,
                               const bool healthy_tail_pass) const
  {
    HypothesisEvidence evidence;
    evidence.range_scale = range_scale;
    evidence.sample_count = size();
    evidence.healthy_tail_pass = healthy_tail_pass;
    evidence.valid = !residuals_.empty();
    if (evidence.valid)
    {
      evidence.residual_median = median(residuals_);
      evidence.match_ratio_median = median(match_ratios_);
      evidence.innovation_median = median(innovations_);
    }
    return evidence;
  }

private:
  static double median(const std::vector<double> &values)
  {
    std::vector<double> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2;
    if (sorted.size() % 2 != 0) return sorted[middle];
    return 0.5 * (sorted[middle - 1] + sorted[middle]);
  }

  std::vector<double> residuals_;
  std::vector<double> match_ratios_;
  std::vector<double> innovations_;
};

struct Estimate
{
  bool accepted = false;
  std::string reason = "not_evaluated";
  double range_scale = 1.0;
  double inverse_range_scale = 1.0;
  double range_scale_lower = 1.0;
  double range_scale_upper = 1.0;
  double confidence = 0.0;
  double winner_score = std::numeric_limits<double>::infinity();
  double identity_score = std::numeric_limits<double>::infinity();
  double relative_improvement = 0.0;
  double runner_up_margin = 0.0;
  std::size_t eligible_hypotheses = 0;
  std::size_t winner_samples = 0;
};

enum class RampStartAnchor
{
  LOWER_BOUND,
  POINT_ESTIMATE,
  UPPER_BOUND
};

struct CorrectionPlan
{
  bool active = false;
  std::string reason = "no_accepted_estimate";
  double replay_start_sec = -1.0;
  double onset_point_sec = -1.0;
  double onset_upper_sec = -1.0;
  double ramp_start_sec = -1.0;
  double ramp_end_sec = -1.0;
  double replay_end_sec = -1.0;
  double terminal_range_scale = 1.0;
  double confidence = 0.0;

  bool appliesAt(const double relative_sec) const
  {
    return active && std::isfinite(relative_sec) &&
        relative_sec >= replay_start_sec && relative_sec <= replay_end_sec;
  }

  double rangeScaleAt(const double relative_sec) const
  {
    if (!appliesAt(relative_sec)) return 1.0;
    if (relative_sec <= ramp_start_sec) return 1.0;
    const double duration = ramp_end_sec - ramp_start_sec;
    if (duration <= 0.0) return terminal_range_scale;
    const double phase = std::max(
        0.0, std::min(1.0, (relative_sec - ramp_start_sec) / duration));
    return 1.0 + phase * (terminal_range_scale - 1.0);
  }

  double inverseScaleAt(const double relative_sec) const
  {
    return 1.0 / rangeScaleAt(relative_sec);
  }
};

class RangeScaleCorrectionEstimator
{
public:
  explicit RangeScaleCorrectionEstimator(const Config &config = Config())
      : config_(config)
  {
    config_.validate();
  }

  Estimate estimate(const OnsetInterval &onset,
                    const std::vector<HypothesisEvidence> &hypotheses) const
  {
    Estimate result;
    if (!config_.enabled)
    {
      result.reason = "disabled";
      return result;
    }
    if (!onset.valid())
    {
      result.reason = "invalid_onset_interval";
      return result;
    }
    if (onset.confidence < config_.minimum_onset_confidence)
    {
      result.reason = "low_onset_confidence";
      return result;
    }
    if (hypotheses.size() < config_.minimum_hypotheses)
    {
      result.reason = "insufficient_hypotheses";
      return result;
    }

    const HypothesisEvidence *identity = nullptr;
    for (const HypothesisEvidence &hypothesis : hypotheses)
      if (validMetrics(hypothesis) &&
          std::abs(hypothesis.range_scale - config_.identity_range_scale) <=
              config_.identity_tolerance)
      {
        identity = &hypothesis;
        break;
      }
    if (identity == nullptr || identity->sample_count < config_.minimum_samples)
    {
      result.reason = "missing_identity_control";
      return result;
    }

    struct Scored
    {
      const HypothesisEvidence *evidence = nullptr;
      double score = std::numeric_limits<double>::infinity();
    };
    std::vector<Scored> scored;
    result.identity_score = score(*identity, *identity);
    for (const HypothesisEvidence &hypothesis : hypotheses)
    {
      if (!validMetrics(hypothesis) ||
          hypothesis.sample_count < config_.minimum_samples)
        continue;
      // Every sample-qualified branch is comparative evidence.  Removing an
      // unhealthy branch before scoring makes a discriminative grid
      // impossible: requiring N eligible hypotheses would also require every
      // deliberately wrong scale to pass the health gate.  Keep all branches
      // for ranking, then require the selected non-identity winner itself to
      // have passed healthy-tail validation.
      scored.push_back({&hypothesis, score(hypothesis, *identity)});
    }
    result.eligible_hypotheses = scored.size();
    if (scored.size() < config_.minimum_hypotheses)
    {
      result.reason = "insufficient_eligible_hypotheses";
      return result;
    }
    // Identity is the conservative null hypothesis.  A non-identity branch is
    // admissible only when the unmodified replay itself fails the independently
    // configured healthy-tail validator.  Otherwise small stochastic score
    // improvements can turn ordinary clean variation into a public correction.
    // This rule consumes no ground truth, clean map, or injected parameters.
    if (identity->healthy_tail_pass)
    {
      result.reason = "healthy_identity_control";
      result.range_scale = config_.identity_range_scale;
      result.inverse_range_scale = 1.0 / config_.identity_range_scale;
      result.range_scale_lower = config_.identity_range_scale;
      result.range_scale_upper = config_.identity_range_scale;
      result.winner_score = result.identity_score;
      result.winner_samples = identity->sample_count;
      return result;
    }
    std::sort(scored.begin(), scored.end(), [this](const Scored &a,
                                                    const Scored &b) {
      if (a.score != b.score) return a.score < b.score;
      const double a_distance = std::abs(
          a.evidence->range_scale - config_.identity_range_scale);
      const double b_distance = std::abs(
          b.evidence->range_scale - config_.identity_range_scale);
      if (a_distance != b_distance) return a_distance < b_distance;
      return a.evidence->range_scale < b.evidence->range_scale;
    });

    const Scored &winner = scored.front();
    result.winner_score = winner.score;
    result.range_scale = winner.evidence->range_scale;
    result.inverse_range_scale = 1.0 / result.range_scale;
    result.winner_samples = winner.evidence->sample_count;
    result.relative_improvement = std::max(
        0.0, (result.identity_score - result.winner_score) /
                 std::max(result.identity_score, 1e-12));
    if (std::abs(result.range_scale - config_.identity_range_scale) <=
        config_.identity_tolerance)
    {
      result.reason = "identity_control_best";
      return result;
    }
    if (!winner.evidence->healthy_tail_pass)
    {
      result.reason = "winner_failed_healthy_tail";
      return result;
    }
    if (result.relative_improvement < config_.minimum_relative_improvement)
    {
      result.reason = "insufficient_improvement";
      return result;
    }

    const double runner_score = scored.size() > 1
        ? scored[1].score
        : result.identity_score;
    result.runner_up_margin = std::max(
        0.0, (runner_score - result.winner_score) /
                 std::max(result.identity_score, 1e-12));
    if (result.runner_up_margin < config_.minimum_runner_up_margin)
    {
      result.reason = "ambiguous_winner";
      return result;
    }

    result.range_scale_lower = result.range_scale;
    result.range_scale_upper = result.range_scale;
    for (const Scored &candidate : scored)
      if (candidate.evidence->healthy_tail_pass &&
          candidate.score <=
              result.winner_score + config_.near_optimal_score_delta)
      {
        result.range_scale_lower = std::min(
            result.range_scale_lower, candidate.evidence->range_scale);
        result.range_scale_upper = std::max(
            result.range_scale_upper, candidate.evidence->range_scale);
      }
    const double interval_width =
        result.range_scale_upper - result.range_scale_lower;
    if (interval_width > config_.maximum_confidence_interval_width)
    {
      result.reason = "wide_scale_confidence_interval";
      return result;
    }

    const double improvement_confidence = clamp01(
        (result.relative_improvement - config_.minimum_relative_improvement) /
        std::max(1e-12, 1.0 - config_.minimum_relative_improvement));
    const double margin_confidence = config_.minimum_runner_up_margin == 0.0
        ? 1.0
        : clamp01(result.runner_up_margin /
                  (3.0 * config_.minimum_runner_up_margin));
    const double sample_confidence = clamp01(
        static_cast<double>(result.winner_samples) /
        static_cast<double>(2 * config_.minimum_samples));
    const double interval_confidence =
        config_.maximum_confidence_interval_width == 0.0
        ? (interval_width == 0.0 ? 1.0 : 0.0)
        : clamp01(1.0 - interval_width /
                            config_.maximum_confidence_interval_width);
    const double evidence_confidence =
        0.35 * improvement_confidence + 0.25 * margin_confidence +
        0.20 * sample_confidence + 0.20 * interval_confidence;
    result.confidence = std::min(onset.confidence, evidence_confidence);
    if (result.confidence < config_.minimum_combined_confidence)
    {
      result.reason = "low_combined_confidence";
      return result;
    }

    result.accepted = true;
    result.reason = "accepted";
    return result;
  }

  CorrectionPlan buildReplayPlan(const OnsetInterval &onset,
                                 const Estimate &estimate) const
  {
    CorrectionPlan plan;
    if (!estimate.accepted)
    {
      plan.reason = estimate.reason;
      return plan;
    }
    if (!onset.valid())
    {
      plan.reason = "invalid_onset_interval";
      return plan;
    }
    plan.active = true;
    plan.reason = "accepted_causal_replay_plan";
    plan.replay_start_sec = onset.lower_bound_sec;
    plan.onset_point_sec = onset.point_estimate_sec;
    plan.onset_upper_sec = onset.upper_bound_sec;
    plan.ramp_start_sec = onset.lower_bound_sec;
    plan.ramp_end_sec = onset.detection_sec;
    plan.replay_end_sec = onset.detection_sec;
    plan.terminal_range_scale = estimate.range_scale;
    plan.confidence = estimate.confidence;
    return plan;
  }

private:
  bool validMetrics(const HypothesisEvidence &hypothesis) const
  {
    return hypothesis.valid && std::isfinite(hypothesis.range_scale) &&
        hypothesis.range_scale >= config_.minimum_range_scale &&
        hypothesis.range_scale <= config_.maximum_range_scale &&
        std::isfinite(hypothesis.residual_median) &&
        hypothesis.residual_median >= 0.0 &&
        std::isfinite(hypothesis.match_ratio_median) &&
        hypothesis.match_ratio_median >= 0.0 &&
        hypothesis.match_ratio_median <= 1.0 &&
        std::isfinite(hypothesis.innovation_median) &&
        hypothesis.innovation_median >= 0.0;
  }

  double score(const HypothesisEvidence &candidate,
               const HypothesisEvidence &identity) const
  {
    const double residual_ratio = boundedRatio(
        candidate.residual_median, identity.residual_median);
    const double candidate_match_error = std::max(
        config_.match_error_floor, 1.0 - candidate.match_ratio_median);
    const double identity_match_error = std::max(
        config_.match_error_floor, 1.0 - identity.match_ratio_median);
    const double match_ratio = std::min(
        config_.ratio_cap, candidate_match_error / identity_match_error);
    const double innovation_ratio = boundedRatio(
        candidate.innovation_median, identity.innovation_median);
    const double weight_sum = config_.residual_weight + config_.match_weight +
                              config_.innovation_weight;
    return (config_.residual_weight * residual_ratio +
            config_.match_weight * match_ratio +
            config_.innovation_weight * innovation_ratio) /
        weight_sum;
  }

  double boundedRatio(const double numerator, const double denominator) const
  {
    // The additive floor keeps the identity/control ratio exactly one even
    // when a metric is numerically zero.  Without it, a perfect control could
    // acquire a spuriously low aggregate score and distort the improvement
    // denominator.
    return std::min(config_.ratio_cap,
                    (numerator + 1e-12) / (denominator + 1e-12));
  }

  static double clamp01(const double value)
  {
    return std::max(0.0, std::min(1.0, value));
  }

  Config config_;
};

// Builds one isolated candidate hypothesis. Selection is intentionally not
// performed here: every candidate is replayed in a private process, its health
// evidence is collected, and RangeScaleCorrectionEstimator subsequently
// decides whether any hypothesis is admissible. The bounded hold after
// detection prevents this development path from consuming an oracle fault-end
// time while still allowing correction of messages arriving after detection.
inline CorrectionPlan makeBoundedCandidateReplayPlan(
    const OnsetInterval &onset, const double terminal_range_scale,
    const double post_detection_horizon_sec,
    const double ramp_end_offset_sec = 0.0,
    const RampStartAnchor ramp_start_anchor = RampStartAnchor::LOWER_BOUND)
{
  CorrectionPlan plan;
  if (!onset.valid())
  {
    plan.reason = "invalid_onset_interval";
    return plan;
  }
  if (!std::isfinite(terminal_range_scale) || terminal_range_scale <= 0.0)
  {
    plan.reason = "invalid_terminal_range_scale";
    return plan;
  }
  if (!std::isfinite(post_detection_horizon_sec) ||
      post_detection_horizon_sec <= 0.0)
  {
    plan.reason = "invalid_post_detection_horizon";
    return plan;
  }
  if (!std::isfinite(ramp_end_offset_sec) || ramp_end_offset_sec < 0.0 ||
      ramp_end_offset_sec > post_detection_horizon_sec)
  {
    plan.reason = "invalid_ramp_end_offset";
    return plan;
  }
  plan.active = true;
  plan.reason = "bounded_candidate_hypothesis";
  plan.replay_start_sec = onset.lower_bound_sec;
  plan.onset_point_sec = onset.point_estimate_sec;
  plan.onset_upper_sec = onset.upper_bound_sec;
  switch (ramp_start_anchor)
  {
    case RampStartAnchor::LOWER_BOUND:
      plan.ramp_start_sec = onset.lower_bound_sec;
      break;
    case RampStartAnchor::POINT_ESTIMATE:
      plan.ramp_start_sec = onset.point_estimate_sec;
      break;
    case RampStartAnchor::UPPER_BOUND:
      plan.ramp_start_sec = onset.upper_bound_sec;
      break;
  }
  // A scale-drift detector can fire while corruption is still increasing.
  // The legacy profile (offset 0) reaches the hypothesized terminal scale at
  // detection.  A positive, causally preconfigured offset instead interprets
  // the same scale as the terminal value at a later point within the bounded
  // observation horizon.  Candidate selection still consumes only online
  // health evidence; no injected fault end is exposed here.
  plan.ramp_end_sec = onset.detection_sec + ramp_end_offset_sec;
  plan.replay_end_sec = onset.detection_sec + post_detection_horizon_sec;
  if (plan.ramp_start_sec < plan.replay_start_sec ||
      plan.ramp_start_sec > plan.ramp_end_sec ||
      plan.ramp_end_sec > plan.replay_end_sec)
  {
    plan.active = false;
    plan.reason = "invalid_candidate_time_order";
    return plan;
  }
  plan.terminal_range_scale = terminal_range_scale;
  plan.confidence = onset.confidence;
  return plan;
}

} // namespace correction_estimation
} // namespace fast_livo

#endif // RANGE_SCALE_CORRECTION_ESTIMATOR_H_
