#include "healthy_tail_validator.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
  fast_livo::recovery_validation::Config config;
  config.enabled = true;
  config.consecutive_healthy = 3;
  config.minimum_healthy_duration_sec = 0.2;
  config.timeout_sec = 1.0;

  fast_livo::recovery_validation::HealthyTailValidator validator(config);
  validator.arm(2.0, 0.010, 0.80);

  auto decision = validator.observe(2.1, 0.020, 0.60);
  assert(!decision.healthy_sample && !decision.approved);
  assert(!decision.residual_pass && !decision.match_ratio_pass);
  assert(std::abs(decision.residual_margin + 0.010) < 1e-12);
  assert(std::abs(decision.match_ratio_margin + 0.20) < 1e-12);
  decision = validator.observe(2.2, 0.008, 0.90);
  assert(decision.healthy_sample && decision.consecutive_healthy == 1);
  decision = validator.observe(2.3, 0.012, 0.90);
  assert(!decision.healthy_sample && decision.consecutive_healthy == 0);
  decision = validator.observe(2.4, 0.008, 0.90);
  decision = validator.observe(2.5, 0.009, 0.88);
  decision = validator.observe(2.6, 0.007, 0.91);
  assert(decision.approved);
  assert(std::abs(decision.healthy_tail_start_sec - 2.4) < 1e-12);
  assert(validator.commitApproved());
  const auto tail = validator.tailDiagnostics();
  assert(tail.observation_count == 6);
  assert(tail.residual_pass_samples == 4);
  assert(tail.match_ratio_pass_samples == 5);
  assert(tail.jointly_healthy_samples == 4);
  assert(tail.longest_consecutive_healthy == 3);
  assert(std::abs(tail.longest_healthy_duration_sec - 0.2) < 1e-12);
  assert(std::abs(tail.minimum_residual_margin + 0.010) < 1e-12);
  assert(std::abs(tail.minimum_match_ratio_margin + 0.20) < 1e-12);
  assert(validator.failedChecks() == "none");

  config.candidate_summary_required = true;
  fast_livo::recovery_validation::HealthyTailValidator summary_validator(config);
  summary_validator.arm(2.0, 0.010, 0.80);
  summary_validator.observe(2.2, 0.008, 0.90);
  summary_validator.observe(2.3, 0.008, 0.90);
  summary_validator.observe(2.41, 0.008, 0.90);
  assert(summary_validator.approved() && !summary_validator.commitApproved());
  summary_validator.setCandidateSummary(0.10, 1.0, 1.05, 0.95, 0.20, 0.50);
  assert(summary_validator.candidateSummaryHealthy());
  assert(summary_validator.commitApproved());
  assert(std::abs(summary_validator.meanObservedMatchRatio() - 0.90) < 1e-12);
  const auto safe_summary = summary_validator.candidateSummaryDiagnostics();
  assert(safe_summary.set && safe_summary.healthy);
  assert(safe_summary.map_support_pass);
  assert(std::abs(safe_summary.map_support_margin - 0.20) < 1e-12);
  assert(summary_validator.diagnosticFields().find(
      "healthy_tail_failed_checks=none") != std::string::npos);

  fast_livo::recovery_validation::HealthyTailValidator unsafe_summary(config);
  unsafe_summary.arm(2.0, 0.010, 0.80);
  unsafe_summary.observe(2.2, 0.008, 0.90);
  unsafe_summary.observe(2.3, 0.008, 0.90);
  unsafe_summary.observe(2.41, 0.008, 0.90);
  unsafe_summary.setCandidateSummary(0.80, 1.0, 1.05, 0.95, 0.20, 0.50);
  assert(!unsafe_summary.candidateSummaryHealthy());
  assert(!unsafe_summary.commitApproved());

  fast_livo::recovery_validation::HealthyTailValidator unsupported_summary(config);
  unsupported_summary.arm(2.0, 0.010, 0.80);
  unsupported_summary.observe(2.2, 0.008, 0.90);
  unsupported_summary.observe(2.3, 0.008, 0.90);
  unsupported_summary.observe(2.41, 0.008, 0.90);
  unsupported_summary.setCandidateSummary(0.10, 1.0, 1.05, 0.60, 0.20, 0.50);
  assert(!unsupported_summary.candidateSummaryHealthy());
  assert(!unsupported_summary.commitApproved());
  assert(unsupported_summary.failedChecks() == "map_support");
  assert(std::abs(unsupported_summary.candidateSummaryDiagnostics().
      map_support_margin + 0.15) < 1e-12);

  fast_livo::recovery_validation::HealthyTailValidator divergent_summary(config);
  divergent_summary.arm(2.0, 0.010, 0.80);
  divergent_summary.observe(2.2, 0.008, 0.90);
  divergent_summary.observe(2.3, 0.008, 0.90);
  divergent_summary.observe(2.41, 0.008, 0.90);
  divergent_summary.setCandidateSummary(0.10, 1.0, 1.05, 0.95, 1.20, 0.50);
  assert(!divergent_summary.candidateSummaryHealthy());
  assert(!divergent_summary.commitApproved());
  assert(divergent_summary.failedChecks() == "boundary_position");

  config.use_map_support_check = false;
  fast_livo::recovery_validation::HealthyTailValidator support_ablation(config);
  support_ablation.arm(2.0, 0.010, 0.80);
  support_ablation.observe(2.2, 0.008, 0.90);
  support_ablation.observe(2.3, 0.008, 0.90);
  support_ablation.observe(2.41, 0.008, 0.90);
  support_ablation.setCandidateSummary(0.10, 1.0, 1.05, 0.60, 0.20, 0.50);
  assert(support_ablation.candidateSummaryHealthy());
  assert(support_ablation.commitApproved());
  assert(std::abs(support_ablation.candidateSummaryDiagnostics().
      map_support_margin + 0.15) < 1e-12);

  config.use_map_support_check = true;
  config.candidate_summary_required = false;
  config.use_residual_check = false;
  fast_livo::recovery_validation::HealthyTailValidator residual_ablation(config);
  residual_ablation.arm(2.0, 0.010, 0.80);
  decision = residual_ablation.observe(2.2, 0.020, 0.90);
  assert(decision.residual_pass && decision.match_ratio_pass &&
         decision.healthy_sample);
  residual_ablation.observe(2.3, 0.020, 0.90);
  decision = residual_ablation.observe(2.41, 0.020, 0.90);
  assert(decision.approved && residual_ablation.commitApproved());
  assert(residual_ablation.tailDiagnostics().residual_pass_samples == 0);

  fast_livo::recovery_validation::HealthyTailValidator timeout_validator(config);
  timeout_validator.arm(5.0, 0.010, 0.80);
  decision = timeout_validator.observe(6.0, 0.020, 0.60);
  assert(decision.timed_out && !decision.approved);

  // P2 sequential mode tolerates isolated bad samples, but approves only once
  // the lower confidence bounds of the complete tail clear every requirement.
  fast_livo::recovery_validation::Config sequential_config;
  sequential_config.enabled = true;
  sequential_config.sequential_evidence_enabled = true;
  sequential_config.minimum_sequential_observations = 10;
  sequential_config.minimum_healthy_duration_sec = 0.9;
  sequential_config.timeout_sec = 2.0;
  sequential_config.confidence_z = 1.0;
  sequential_config.minimum_residual_pass_rate_lower_bound = 0.60;
  sequential_config.minimum_match_pass_rate_lower_bound = 0.60;
  sequential_config.minimum_joint_pass_rate_lower_bound = 0.50;
  fast_livo::recovery_validation::HealthyTailValidator sequential_validator(
      sequential_config);
  sequential_validator.arm(10.0, 0.010, 0.80);
  for (int i = 0; i < 10; ++i)
  {
    const bool isolated_outlier = i == 4;
    decision = sequential_validator.observe(
        10.0 + 0.1 * i,
        isolated_outlier ? 0.020 : 0.008,
        isolated_outlier ? 0.60 : 0.90);
  }
  assert(decision.approved && decision.sequential_evidence_pass);
  assert(decision.consecutive_healthy == 0);
  const auto sequential_tail = sequential_validator.tailDiagnostics();
  assert(sequential_tail.sequential_evidence_enabled);
  assert(sequential_tail.sequential_evidence_pass);
  assert(sequential_tail.residual_pass_samples == 9);
  assert(sequential_tail.match_ratio_pass_samples == 9);
  assert(sequential_tail.jointly_healthy_samples == 9);
  assert(sequential_tail.residual_pass_rate_lower_bound > 0.60);
  assert(sequential_tail.match_pass_rate_lower_bound > 0.60);
  assert(sequential_tail.joint_pass_rate_lower_bound > 0.50);
  assert(sequential_validator.localHealthApproved());
  assert(std::abs(sequential_validator.healthyTailStartTime() - 10.0) < 1e-12);
  assert(sequential_validator.diagnosticFields().find(
      "healthy_tail_sequential_evidence_pass=1") != std::string::npos);
  assert(sequential_validator.diagnosticFields().find(
      "healthy_tail_sequential_observation_margin_samples=0") !=
      std::string::npos);

  // Approval cannot latch on an early healthy prefix: the final decision must
  // represent the complete post-quarantine evidence available at commit time.
  for (int i = 10; i < 20; ++i)
    decision = sequential_validator.observe(
        10.0 + 0.1 * i, 0.020, 0.60);
  assert(!decision.approved && !decision.sequential_evidence_pass);
  assert(!sequential_validator.localHealthApproved());

  fast_livo::recovery_validation::HealthyTailValidator noisy_validator(
      sequential_config);
  noisy_validator.arm(20.0, 0.010, 0.80);
  for (int i = 0; i < 10; ++i)
    decision = noisy_validator.observe(
        20.0 + 0.1 * i, i < 5 ? 0.008 : 0.020,
        i < 5 ? 0.90 : 0.60);
  assert(!decision.approved && !decision.sequential_evidence_pass);
  assert(noisy_validator.failedChecks() ==
         "healthy_tail_sequential_confidence");

  std::cout << "HEALTHY_TAIL_VALIDATOR_TEST_OK" << std::endl;
  return 0;
}
