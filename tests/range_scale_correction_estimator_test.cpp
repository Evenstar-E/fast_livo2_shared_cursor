#include "range_scale_correction_estimator.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

using fast_livo::correction_estimation::Config;
using fast_livo::correction_estimation::HypothesisEvidence;
using fast_livo::correction_estimation::HypothesisHistory;
using fast_livo::correction_estimation::OnsetInterval;
using fast_livo::correction_estimation::RangeScaleCorrectionEstimator;
using fast_livo::correction_estimation::RampStartAnchor;
using fast_livo::correction_estimation::makeBoundedCandidateReplayPlan;

Config enabledConfig()
{
  Config config;
  config.enabled = true;
  config.minimum_samples = 20;
  config.minimum_hypotheses = 3;
  config.minimum_onset_confidence = 0.60;
  config.minimum_relative_improvement = 0.10;
  config.minimum_runner_up_margin = 0.04;
  config.minimum_combined_confidence = 0.60;
  return config;
}

OnsetInterval validOnset()
{
  OnsetInterval onset;
  onset.lower_bound_sec = 80.0;
  onset.point_estimate_sec = 80.5;
  onset.upper_bound_sec = 81.0;
  onset.confidence = 0.88;
  onset.detection_sec = 84.0;
  return onset;
}

HypothesisEvidence evidence(const double scale, const double residual,
                            const double match, const double innovation,
                            const bool healthy_tail = true,
                            const std::size_t samples = 60)
{
  HypothesisEvidence value;
  value.range_scale = scale;
  value.residual_median = residual;
  value.match_ratio_median = match;
  value.innovation_median = innovation;
  value.sample_count = samples;
  value.healthy_tail_pass = healthy_tail;
  return value;
}

} // namespace

int main()
{
  Config config = enabledConfig();
  RangeScaleCorrectionEstimator estimator(config);

  HypothesisHistory history;
  history.observe(0.03, 0.80, 1.2);
  history.observe(0.01, 0.95, 0.4);
  history.observe(0.02, 0.90, 0.8);
  history.observe(-1.0, 0.90, 0.8); // Invalid observations are not evidence.
  const HypothesisEvidence summary = history.summarize(1.04, true);
  assert(summary.sample_count == 3);
  assert(std::abs(summary.residual_median - 0.02) < 1e-12);
  assert(std::abs(summary.match_ratio_median - 0.90) < 1e-12);
  assert(std::abs(summary.innovation_median - 0.8) < 1e-12);

  const std::vector<HypothesisEvidence> recoverable = {
      evidence(1.00, 0.040, 0.70, 2.0, false),
      evidence(1.02, 0.025, 0.82, 1.2),
      evidence(1.04, 0.010, 0.95, 0.4),
      evidence(1.06, 0.018, 0.89, 0.8)};
  const auto accepted = estimator.estimate(validOnset(), recoverable);
  assert(accepted.accepted);
  assert(accepted.reason == "accepted");
  assert(std::abs(accepted.range_scale - 1.04) < 1e-12);
  assert(std::abs(accepted.inverse_range_scale - 1.0 / 1.04) < 1e-12);
  assert(accepted.range_scale_lower == accepted.range_scale_upper);
  assert(accepted.relative_improvement > 0.70);
  assert(accepted.confidence >= config.minimum_combined_confidence);

  const auto plan = estimator.buildReplayPlan(validOnset(), accepted);
  assert(plan.active);
  assert(std::abs(plan.ramp_start_sec - 80.0) < 1e-12);
  assert(!plan.appliesAt(79.999));
  assert(plan.appliesAt(80.0));
  assert(plan.appliesAt(84.0));
  assert(!plan.appliesAt(84.001));
  assert(std::abs(plan.rangeScaleAt(80.0) - 1.0) < 1e-12);
  assert(std::abs(plan.rangeScaleAt(82.0) - 1.02) < 1e-12);
  assert(std::abs(plan.rangeScaleAt(84.0) - 1.04) < 1e-12);
  assert(std::abs(1.04 * plan.inverseScaleAt(84.0) - 1.0) < 1e-12);

  const auto bounded = makeBoundedCandidateReplayPlan(
      validOnset(), 1.04, 8.0);
  assert(bounded.active);
  assert(bounded.appliesAt(92.0));
  assert(!bounded.appliesAt(92.001));
  assert(std::abs(bounded.rangeScaleAt(82.0) - 1.02) < 1e-12);
  assert(std::abs(bounded.rangeScaleAt(88.0) - 1.04) < 1e-12);

  // A continuing-drift hypothesis reaches its terminal value at the end of
  // the causal observation horizon, rather than prematurely at detection.
  const auto continuing_drift = makeBoundedCandidateReplayPlan(
      validOnset(), 1.04, 8.0, 8.0);
  assert(continuing_drift.active);
  assert(std::abs(continuing_drift.ramp_end_sec - 92.0) < 1e-12);
  assert(std::abs(continuing_drift.rangeScaleAt(84.0) -
                  (1.0 + 0.04 / 3.0)) < 1e-12);
  assert(std::abs(continuing_drift.rangeScaleAt(88.0) -
                  (1.0 + 0.08 / 3.0)) < 1e-12);
  assert(std::abs(continuing_drift.rangeScaleAt(92.0) - 1.04) < 1e-12);

  // Roll back from the conservative lower bound, while leaving the input
  // untouched until the upper confidence bound. This removes any already
  // committed contamination without correcting the uncertain clean prefix.
  const auto upper_anchored = makeBoundedCandidateReplayPlan(
      validOnset(), 1.04, 8.0, 8.0, RampStartAnchor::UPPER_BOUND);
  assert(upper_anchored.active);
  assert(std::abs(upper_anchored.replay_start_sec - 80.0) < 1e-12);
  assert(std::abs(upper_anchored.ramp_start_sec - 81.0) < 1e-12);
  assert(std::abs(upper_anchored.rangeScaleAt(80.5) - 1.0) < 1e-12);
  assert(std::abs(upper_anchored.rangeScaleAt(81.0) - 1.0) < 1e-12);
  assert(std::abs(upper_anchored.rangeScaleAt(86.5) - 1.02) < 1e-12);
  assert(std::abs(upper_anchored.rangeScaleAt(92.0) - 1.04) < 1e-12);

  const auto invalid_profile = makeBoundedCandidateReplayPlan(
      validOnset(), 1.04, 8.0, 8.001);
  assert(!invalid_profile.active);
  assert(invalid_profile.reason == "invalid_ramp_end_offset");

  // Clean/no-op gate: if identity is best, no correction may be admitted.
  const std::vector<HypothesisEvidence> clean = {
      evidence(1.00, 0.010, 0.97, 0.3),
      evidence(1.02, 0.014, 0.94, 0.5),
      evidence(1.04, 0.020, 0.90, 0.8)};
  const auto clean_result = estimator.estimate(validOnset(), clean);
  assert(!clean_result.accepted);
  assert(clean_result.reason == "healthy_identity_control");
  assert(clean_result.range_scale == 1.0);

  const std::vector<HypothesisEvidence> perfect_clean = {
      evidence(1.00, 0.0, 1.0, 0.0),
      evidence(1.02, 0.001, 0.99, 0.1),
      evidence(1.04, 0.002, 0.98, 0.2)};
  const auto perfect_clean_result = estimator.estimate(
      validOnset(), perfect_clean);
  assert(!perfect_clean_result.accepted);
  assert(perfect_clean_result.reason == "healthy_identity_control");
  assert(std::abs(perfect_clean_result.identity_score - 1.0) < 1e-12);

  // Regression from the clean exp16 selector control: 1.03 had a lower scalar
  // score, but the identity replay was itself healthy and must retain the
  // public incumbent.
  const std::vector<HypothesisEvidence> observed_clean = {
      evidence(1.00, 0.00803311460431373, 0.920567375886525,
               1.46464418727751, true, 41),
      evidence(1.01, 0.0212349382568123, 0.130252100840336,
               2.19871205804679, false, 41),
      evidence(1.02, 0.0095254860248479, 0.886740331491713,
               1.90263518244534, false, 41),
      evidence(1.03, 0.00735679801585323, 0.905686546463246,
               0.624163864819643, true, 41)};
  const auto observed_clean_result = estimator.estimate(
      validOnset(), observed_clean);
  assert(!observed_clean_result.accepted);
  assert(observed_clean_result.reason == "healthy_identity_control");
  assert(observed_clean_result.range_scale == 1.0);

  // Regression from the fault exp16 selector control: identity is unhealthy,
  // so the same conservative rule still allows the healthy 1.03 correction.
  const std::vector<HypothesisEvidence> observed_fault = {
      evidence(1.00, 0.0227210718479416, 0.149586776859504,
               2.25261463504556, false, 41),
      evidence(1.01, 0.03871939266029975, 0.0211416595482628,
               1.17127361620514, false, 40),
      evidence(1.02, 0.0290259050181538, 0.0411347517730496,
               1.56999989037266, false, 41),
      evidence(1.03, 0.00821545114522313, 0.903818953323904,
               1.2127473379636, true, 41)};
  const auto observed_fault_result = estimator.estimate(
      validOnset(), observed_fault);
  assert(observed_fault_result.accepted);
  assert(observed_fault_result.reason == "accepted");
  assert(std::abs(observed_fault_result.range_scale - 1.03) < 1e-12);

  // Closely tied candidate evidence must fail closed.
  const std::vector<HypothesisEvidence> ambiguous = {
      evidence(1.00, 0.040, 0.70, 2.0, false),
      evidence(1.03, 0.0110, 0.945, 0.42),
      evidence(1.04, 0.0109, 0.946, 0.41)};
  const auto ambiguous_result = estimator.estimate(validOnset(), ambiguous);
  assert(!ambiguous_result.accepted);
  assert(ambiguous_result.reason == "ambiguous_winner");

  OnsetInterval low_confidence = validOnset();
  low_confidence.confidence = 0.30;
  const auto low_onset = estimator.estimate(low_confidence, recoverable);
  assert(!low_onset.accepted);
  assert(low_onset.reason == "low_onset_confidence");

  const std::vector<HypothesisEvidence> no_identity = {
      evidence(1.02, 0.025, 0.82, 1.2),
      evidence(1.04, 0.010, 0.95, 0.4),
      evidence(1.06, 0.018, 0.89, 0.8)};
  const auto missing_control = estimator.estimate(validOnset(), no_identity);
  assert(!missing_control.accepted);
  assert(missing_control.reason == "missing_identity_control");

  // Merely supplying rows is not sufficient: the selector still needs the
  // configured number of independently sample-qualified hypotheses.
  const std::vector<HypothesisEvidence> insufficient_eligible = {
      evidence(1.00, 0.040, 0.70, 2.0, false),
      evidence(1.02, 0.025, 0.82, 1.2, false, 5),
      evidence(1.04, 0.010, 0.95, 0.4),
      evidence(1.06, 0.018, 0.89, 0.8, true, 5)};
  const auto insufficient = estimator.estimate(
      validOnset(), insufficient_eligible);
  assert(!insufficient.accepted);
  assert(insufficient.reason == "insufficient_eligible_hypotheses");
  assert(insufficient.eligible_hypotheses == 2);

  // Unhealthy alternatives remain useful negative controls, but an unhealthy
  // branch can never be selected even when its scalar score is best.
  const std::vector<HypothesisEvidence> unhealthy_winner = {
      evidence(1.00, 0.040, 0.70, 2.0, false),
      evidence(1.02, 0.004, 0.99, 0.1, false),
      evidence(1.04, 0.010, 0.95, 0.4, true)};
  const auto unhealthy_result = estimator.estimate(
      validOnset(), unhealthy_winner);
  assert(!unhealthy_result.accepted);
  assert(unhealthy_result.reason == "winner_failed_healthy_tail");

  Config invalid = config;
  invalid.minimum_range_scale = invalid.maximum_range_scale;
  bool invalid_rejected = false;
  try
  {
    RangeScaleCorrectionEstimator invalid_estimator(invalid);
  }
  catch (const std::invalid_argument &)
  {
    invalid_rejected = true;
  }
  assert(invalid_rejected);

  std::cout << "RANGE_SCALE_CORRECTION_ESTIMATOR_TEST_OK" << std::endl;
  return 0;
}
