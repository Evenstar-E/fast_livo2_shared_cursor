#include "online_harm_detector.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
fast_livo::harm_detection::OnlineHarmDetector makeDetector()
{
  fast_livo::harm_detection::Config config;
  config.enabled = true;
  config.baseline_window = 20;
  config.trigger_consecutive = 3;
  config.minimum_anomaly_duration_sec = 0.15;
  return fast_livo::harm_detection::OnlineHarmDetector(config);
}

void seedHealthy(fast_livo::harm_detection::OnlineHarmDetector &detector)
{
  for (int i = 0; i < 20; ++i)
  {
    const double residual = 0.006 + 0.0002 * std::sin(i);
    const double match = 0.92 + 0.01 * std::cos(i);
    assert(!detector.observe(0.1 * i, residual, match).triggered);
  }
  assert(detector.baselineSize() == 20);
}

void testAbruptFallbackGuard()
{
  auto detector = makeDetector();
  seedHealthy(detector);

  auto decision = detector.observe(2.0, 0.016, 0.65);
  assert(decision.baseline_ready && decision.anomalous && !decision.triggered);
  decision = detector.observe(2.1, 0.017, 0.63);
  assert(decision.anomalous && !decision.triggered);
  decision = detector.observe(2.2, 0.018, 0.61);
  assert(decision.triggered);
  assert(std::abs(decision.estimated_onset_sec - 2.0) < 1e-12);
  // The adaptive bracket starts at the abrupt strong evidence, but the
  // conservative fixed guard remains the safer lower bound.
  assert(decision.change_point_found);
  assert(!decision.precursor_supported);
  assert(std::abs(decision.change_point_upper_sec - 2.0) < 1e-12);
  assert(std::abs(decision.onset_uncertainty_sec - 0.5) < 1e-12);
  assert(std::abs(decision.onset_lower_bound_sec - 1.5) < 1e-12);
  assert(std::abs(decision.onset_upper_bound_sec - 2.0) < 1e-12);
  assert(decision.onset_confidence > 0.0 && decision.onset_confidence <= 1.0);
  assert(decision.episode_id == 1);
  assert(decision.proposal_index_in_episode == 1);
  assert(decision.residual_median > 0.0);
  assert(decision.match_ratio_median > 0.0);

  decision = detector.observe(2.3, 0.006, 0.92);
  assert(!decision.triggered && detector.triggered());
}

void testFrozenModeSuppressesNaturalTrigger()
{
  auto detector = makeDetector();
  seedHealthy(detector);
  fast_livo::harm_detection::Decision decision;
  for (int i = 0; i < 10; ++i)
  {
    decision = detector.observe(2.0 + 0.1 * i, 0.030, 0.55, 3.0,
                                false);
    assert(decision.valid_observation && decision.baseline_ready);
    assert(!decision.triggered && !detector.triggered());
  }
  decision = detector.observe(3.0, 0.030, 0.55, 3.0);
  assert(decision.triggered);

  fast_livo::harm_detection::Config config;
  config.enabled = true;
  config.two_timescale_enabled = true;
  config.reference_window_sec = 4.0;
  config.reference_gap_sec = 0.0;
  config.current_window_sec = 1.0;
  config.minimum_anomaly_duration_sec = 0.2;
  config.trigger_consecutive = 3;
  config.strong_fused_score = 0.1;
  config.minimum_strong_signals = 2;
  config.two_timescale_minimum_residual_anchor_score = 0.05;
  config.two_timescale_signal_support_score = 0.1;
  fast_livo::harm_detection::OnlineHarmDetector two_scale(config);
  for (int i = 0; i < 200; ++i)
    two_scale.observe(0.1 * i, 0.006, 0.92, 1.0, false);
  for (int i = 200; i < 260; ++i)
  {
    decision = two_scale.observe(0.1 * i, 0.024, 0.62, 3.0, false);
    assert(!decision.triggered && !two_scale.triggered());
  }
  assert(decision.baseline_ready);
}

void testArmingDelayExcludesStartupTransition()
{
  fast_livo::harm_detection::Config config;
  config.enabled = true;
  config.arming_delay_sec = 2.0;
  config.baseline_window = 20;
  config.trigger_consecutive = 3;
  config.minimum_anomaly_duration_sec = 0.15;
  fast_livo::harm_detection::OnlineHarmDetector detector(config);

  for (int i = 0; i < 20; ++i)
  {
    const auto decision = detector.observe(0.1 * i, 0.030, 0.50, 4.0);
    assert(!decision.triggered);
    assert(detector.baselineSize() == 0);
  }
  for (int i = 0; i < 20; ++i)
  {
    const auto decision = detector.observe(2.0 + 0.1 * i, 0.006, 0.92, 0.8);
    assert(!decision.triggered);
  }
  assert(detector.baselineSize() == 20);
  auto decision = detector.observe(4.0, 0.020, 0.62, 3.0);
  decision = detector.observe(4.1, 0.020, 0.62, 3.0);
  decision = detector.observe(4.2, 0.020, 0.62, 3.0);
  assert(decision.triggered);
  assert(decision.onset_lower_bound_sec >= config.arming_delay_sec);
}

void testGradualWeakEvidenceExtendsOnset()
{
  auto detector = makeDetector();
  seedHealthy(detector);

  // Residual-only weak evidence precedes the joint strong trigger by a full
  // second.  No injector label is supplied to the detector.
  for (int i = 0; i < 10; ++i)
  {
    const auto decision = detector.observe(2.0 + 0.1 * i, 0.009, 0.92);
    assert(decision.weak_evidence);
    assert(decision.weak_evidence_samples > 0);
    assert(!decision.triggered);
  }

  // Ten weak samples intentionally broaden the adaptive residual MAD. Keep
  // the strong triplet above that broadened threshold so this Debug-build
  // assertion tests change-point behavior rather than an accidental margin.
  auto decision = detector.observe(3.0, 0.040, 0.65);
  decision = detector.observe(3.1, 0.045, 0.63);
  decision = detector.observe(3.2, 0.050, 0.61);
  assert(decision.triggered);
  assert(decision.change_point_found);
  assert(decision.precursor_supported);
  assert(std::abs(decision.change_point_lower_sec - 1.9) < 1e-12);
  assert(std::abs(decision.change_point_upper_sec - 2.0) < 1e-12);
  assert(decision.weak_evidence_samples >= 13);
  assert(std::abs(decision.estimated_onset_sec - 2.0) < 1e-12);
  assert(std::abs(decision.onset_lower_bound_sec - 1.4) < 1e-12);
  assert(std::abs(decision.onset_upper_bound_sec - 2.0) < 1e-12);
  assert(std::abs(decision.onset_uncertainty_sec - 0.6) < 1e-12);
}

void testIsolatedWeakOutlierDoesNotDragOnset()
{
  auto detector = makeDetector();
  seedHealthy(detector);

  auto weak = detector.observe(2.0, 0.009, 0.92);  // isolated weak sample
  assert(weak.weak_evidence_samples == 1);
  auto one_gap = detector.observe(2.1, 0.006, 0.92);
  assert(one_gap.weak_evidence_samples == 1);
  auto two_gaps = detector.observe(2.2, 0.006, 0.92);  // breaks the suffix
  assert(two_gaps.weak_evidence_samples == 0);
  detector.observe(2.3, 0.006, 0.92);
  auto decision = detector.observe(2.4, 0.016, 0.65);
  decision = detector.observe(2.5, 0.017, 0.63);
  decision = detector.observe(2.6, 0.018, 0.61);
  assert(decision.triggered);
  assert(!decision.precursor_supported);
  assert(decision.change_point_lower_sec >= 2.3 - 1e-12);
  assert(decision.onset_lower_bound_sec >= 1.9 - 1e-12);
}

void testWeakSceneChangeStillAdaptsBaseline()
{
  auto detector = makeDetector();
  seedHealthy(detector);

  // Sustained sub-trigger evidence can also be a benign scene-regime change.
  // It must remain in the adaptive baseline or the detector eventually turns
  // normal operation into a false strong alarm.
  fast_livo::harm_detection::Decision decision;
  for (int i = 0; i < 40; ++i)
  {
    decision = detector.observe(2.0 + 0.1 * i, 0.009, 0.82);
    assert(!decision.triggered);
  }
  decision = detector.observe(6.0, 0.009, 0.82);
  assert(decision.baseline_ready);
  assert(decision.residual_median > 0.0085);
  assert(decision.match_ratio_median < 0.83);
  assert(!decision.anomalous);
}

void testBriefJointSceneChangeDoesNotTrigger()
{
  fast_livo::harm_detection::Config config;
  config.enabled = true;
  config.baseline_window = 20;
  config.trigger_consecutive = 3;
  config.minimum_anomaly_duration_sec = 0.5;
  fast_livo::harm_detection::OnlineHarmDetector detector(config);
  seedHealthy(detector);

  for (int i = 0; i < 4; ++i)
  {
    const auto decision = detector.observe(2.0 + 0.1 * i, 0.018, 0.61);
    assert(decision.anomalous);
    assert(!decision.triggered);
  }
  const auto recovered = detector.observe(2.4, 0.006, 0.92);
  assert(!recovered.anomalous);
  assert(!detector.triggered());
}

void testRearmRequiresFreshHealthyBaseline()
{
  auto detector = makeDetector();
  seedHealthy(detector);
  detector.observe(2.0, 0.016, 0.65);
  detector.observe(2.1, 0.017, 0.63);
  assert(detector.observe(2.2, 0.018, 0.61).triggered);
  detector.resetForMonitoring();
  assert(!detector.triggered());
  assert(detector.baselineSize() == 0);
  for (int i = 0; i < 20; ++i)
    assert(!detector.observe(3.0 + 0.1 * i, 0.009, 0.82).triggered);
  const auto ready = detector.observe(5.0, 0.009, 0.82);
  assert(ready.baseline_ready);
  assert(!ready.anomalous);
}

void testResidualInnovationFusionAndEpisodeMerge()
{
  auto detector = makeDetector();
  seedHealthy(detector);
  auto decision = detector.observe(2.0, 0.016, 0.92, 2.0);
  decision = detector.observe(2.1, 0.017, 0.92, 2.1);
  decision = detector.observe(2.2, 0.018, 0.92, 2.2);
  assert(decision.triggered);
  assert(decision.residual_score > 1.0);
  assert(decision.match_score < 1.0);
  assert(decision.innovation_score > 1.0);
  assert(decision.strong_signal_count >= 2);
  assert(decision.episode_id == 1);
  assert(!decision.merged_with_previous_proposal);

  detector.resetForMonitoring();
  for (int i = 0; i < 20; ++i)
    detector.observe(3.0 + 0.1 * i, 0.006, 0.92, 0.1);
  detector.observe(5.0, 0.016, 0.92, 2.0);
  detector.observe(5.1, 0.017, 0.92, 2.1);
  decision = detector.observe(5.2, 0.018, 0.92, 2.2);
  assert(decision.triggered);
  assert(decision.episode_id == 1);
  assert(decision.proposal_index_in_episode == 2);
  assert(decision.merged_with_previous_proposal);
  assert(decision.episode_lower_bound_sec <= decision.onset_lower_bound_sec);
}

void testInvalidObservationIsSkippedWithoutChangingBaseline()
{
  auto detector = makeDetector();
  auto decision = detector.observe(
      0.0, std::numeric_limits<double>::infinity(), 0.5);
  assert(!decision.valid_observation);
  assert(detector.baselineSize() == 0);
  decision = detector.observe(0.1, 0.006, 1.1);
  assert(!decision.valid_observation);
  assert(detector.baselineSize() == 0);
  decision = detector.observe(0.2, 0.006, 0.92);
  assert(decision.valid_observation);
  assert(detector.baselineSize() == 1);
  decision = detector.observe(
      0.3, 0.006, 0.92, std::numeric_limits<double>::quiet_NaN());
  assert(!decision.valid_observation);
  assert(detector.baselineSize() == 1);
}

void testTwoTimescaleRejectsBriefTransitionAndReportsInterval()
{
  fast_livo::harm_detection::Config config;
  config.enabled = true;
  config.two_timescale_enabled = true;
  config.reference_window_sec = 4.0;
  config.reference_gap_sec = 0.2;
  config.current_window_sec = 1.0;
  config.minimum_anomaly_duration_sec = 1.2;
  config.trigger_consecutive = 3;
  fast_livo::harm_detection::OnlineHarmDetector detector(config);

  fast_livo::harm_detection::Decision decision;
  for (int i = 0; i < 300; ++i)
  {
    const double time = 0.1 * i;
    const bool brief_scene_change = time >= 10.0 && time <= 10.8;
    decision = detector.observe(
        time, 0.006 + (brief_scene_change ? 0.012 : 0.0),
        0.92 - (brief_scene_change ? 0.22 : 0.0),
        0.8 + (brief_scene_change ? 1.5 : 0.0));
    assert(!decision.triggered);
  }

  for (int i = 300; i < 450 && !decision.triggered; ++i)
  {
    const double time = 0.1 * i;
    decision = detector.observe(time, 0.024, 0.62, 3.0);
  }
  assert(decision.triggered);
  assert(decision.strong_signal_count == 3);
  assert(decision.onset_lower_bound_sec <= decision.estimated_onset_sec);
  assert(decision.estimated_onset_sec <= decision.onset_upper_bound_sec);
  assert(std::abs(decision.onset_upper_bound_sec -
                  decision.onset_lower_bound_sec - 1.0) < 1e-9);
  assert(decision.onset_confidence > 0.0 && decision.onset_confidence <= 1.0);
}

void testTwoTimescaleWeakTriggerWidensOnlyRecoveryLeftBoundary()
{
  fast_livo::harm_detection::Config config;
  config.enabled = true;
  config.two_timescale_enabled = true;
  config.reference_window_sec = 4.0;
  config.reference_gap_sec = 0.0;
  config.current_window_sec = 1.0;
  config.minimum_anomaly_duration_sec = 0.2;
  config.trigger_consecutive = 3;
  config.strong_fused_score = 0.1;
  config.minimum_strong_signals = 2;
  config.two_timescale_minimum_residual_anchor_score = 0.05;
  config.two_timescale_signal_support_score = 0.1;
  config.two_timescale_conservative_weak_onset_enabled = true;
  config.two_timescale_conservative_weak_score_threshold = 1.0;
  fast_livo::harm_detection::OnlineHarmDetector detector(config);

  fast_livo::harm_detection::Decision decision;
  for (int i = 0; i < 200; ++i)
    decision = detector.observe(0.1 * i, 0.006, 0.92, 1.0);
  for (int i = 200; i < 260 && !decision.triggered; ++i)
    decision = detector.observe(0.1 * i, 0.0066, 0.90, 1.1);

  assert(decision.triggered);
  assert(decision.residual_score < 1.0);
  assert(decision.match_score < 1.0);
  assert(decision.innovation_score < 1.0);
  assert(decision.onset_interval_semantics ==
         "weak_evidence_causal_support_envelope");
  assert(decision.onset_upper_bound_sec - decision.onset_lower_bound_sec >
         config.current_window_sec + config.reference_window_sec - 0.2);
  // The point estimate remains the nominal current-window midpoint; only the
  // recovery-safe lower boundary is widened.
  assert(decision.estimated_onset_sec > decision.onset_lower_bound_sec);
  assert(decision.estimated_onset_sec <= decision.onset_upper_bound_sec);
}

void testSuppressedProposalRequiresHealthyTransitionBeforeRearm()
{
  fast_livo::harm_detection::Config config;
  config.enabled = true;
  config.two_timescale_enabled = true;
  config.reference_window_sec = 4.0;
  config.reference_gap_sec = 0.0;
  config.current_window_sec = 1.0;
  config.minimum_anomaly_duration_sec = 0.2;
  config.trigger_consecutive = 3;
  config.strong_fused_score = 0.1;
  config.minimum_strong_signals = 2;
  config.two_timescale_minimum_residual_anchor_score = 0.05;
  config.two_timescale_signal_support_score = 0.1;
  fast_livo::harm_detection::OnlineHarmDetector detector(config);

  fast_livo::harm_detection::Decision decision;
  for (int i = 0; i < 200; ++i)
    decision = detector.observe(0.1 * i, 0.006, 0.92, 1.0);
  for (int i = 200; i < 260 && !decision.triggered; ++i)
    decision = detector.observe(0.1 * i, 0.024, 0.62, 3.0);
  assert(decision.triggered);

  detector.suppressUntilHealthyTransition();
  assert(!detector.triggered());
  assert(detector.awaitingHealthyRearm());
  for (int i = 260; i < 300; ++i)
  {
    decision = detector.observe(0.1 * i, 0.024, 0.62, 3.0);
    assert(!decision.triggered);
  }
  assert(detector.awaitingHealthyRearm());

  for (int i = 300; i < 380; ++i)
  {
    decision = detector.observe(0.1 * i, 0.006, 0.92, 1.0);
    assert(!decision.triggered);
  }
  assert(!detector.awaitingHealthyRearm());

  for (int i = 380; i < 460 && !decision.triggered; ++i)
    decision = detector.observe(0.1 * i, 0.024, 0.62, 3.0);
  assert(decision.triggered);
}
} // namespace

int main()
{
  testAbruptFallbackGuard();
  testFrozenModeSuppressesNaturalTrigger();
  testArmingDelayExcludesStartupTransition();
  testGradualWeakEvidenceExtendsOnset();
  testIsolatedWeakOutlierDoesNotDragOnset();
  testWeakSceneChangeStillAdaptsBaseline();
  testBriefJointSceneChangeDoesNotTrigger();
  testRearmRequiresFreshHealthyBaseline();
  testResidualInnovationFusionAndEpisodeMerge();
  testInvalidObservationIsSkippedWithoutChangingBaseline();
  testTwoTimescaleRejectsBriefTransitionAndReportsInterval();
  testTwoTimescaleWeakTriggerWidensOnlyRecoveryLeftBoundary();
  testSuppressedProposalRequiresHealthyTransitionBeforeRearm();
  std::cout << "ONLINE_HARM_DETECTOR_TEST_OK" << std::endl;
  return 0;
}
