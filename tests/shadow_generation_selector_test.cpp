#include "shadow_generation_selector.h"

#include <cassert>
#include <cmath>
#include <iostream>

namespace sm = fast_livo::shadow_mux;

sm::Pose pose(double time, double x, double yaw_deg = 0.0)
{
  sm::Pose value;
  value.time = time;
  value.x = x;
  const double half_yaw = yaw_deg * 3.14159265358979323846 / 360.0;
  value.qz = std::sin(half_yaw);
  value.qw = std::cos(half_yaw);
  value.frame_id = "camera_init";
  value.child_frame_id = "aft_mapped";
  return value;
}

int main()
{
  sm::Config config;
  config.probation_duration_sec = 2.0;
  config.minimum_matched_samples = 3;
  config.minimum_post_catchup_duration_sec = 1.0;
  config.minimum_post_catchup_samples = 2;

  sm::Selector stable(config);
  stable.candidateCommitted();
  stable.observe(pose(10.0, 0.0), pose(10.0, 0.1), 12.1);
  stable.observe(pose(11.0, 1.0), pose(11.0, 1.11), 12.1);
  const sm::Evaluation &waiting =
      stable.observe(pose(12.0, 2.0), pose(12.0, 2.12), 12.1);
  assert(waiting.state == sm::State::PROBATION);
  assert(waiting.reason == "candidate_not_caught_up");
  const sm::Evaluation &tail_waiting =
      stable.observe(pose(12.1, 2.1), pose(12.1, 2.22), 12.1);
  assert(tail_waiting.state == sm::State::PROBATION);
  assert(tail_waiting.reason == "waiting_for_post_catchup_tail");
  assert(tail_waiting.post_catchup_samples == 1);
  const sm::Evaluation &approved =
      stable.observe(pose(13.1, 3.1), pose(13.1, 3.22), 13.1);
  assert(approved.state == sm::State::CANDIDATE);
  assert(approved.reason == "healthy_tail_approved");
  assert(approved.uncertainty_samples == 5);
  assert(std::abs(approved.mean_position_variance_ratio - 1.0) < 1e-12);
  assert(std::abs(approved.mean_rotation_variance_ratio - 1.0) < 1e-12);
  assert(std::abs(approved.second_maximum_position_variance_ratio - 1.0) <
         1e-12);
  assert(std::abs(approved.second_maximum_rotation_variance_ratio - 1.0) <
         1e-12);

  sm::Selector terminal_only(config);
  terminal_only.candidateCommitted();
  terminal_only.observe(pose(20.0, 0.0), pose(20.0, 0.1), 22.1);
  terminal_only.observe(pose(21.0, 1.0), pose(21.0, 1.1), 22.1);
  terminal_only.observe(pose(22.1, 2.0), pose(22.1, 2.1), 22.1);
  assert(terminal_only.finalize().state == sm::State::ABORTED);
  assert(terminal_only.evaluation().reason == "terminal_only_handover");

  sm::Selector never_caught_up(config);
  never_caught_up.candidateCommitted();
  never_caught_up.observe(pose(30.0, 0.0), pose(30.0, 0.1), 40.0);
  assert(never_caught_up.finalize().state == sm::State::ABORTED);
  assert(never_caught_up.evaluation().reason ==
         "candidate_not_caught_up_at_end");

  sm::Selector boundary_bad(config);
  boundary_bad.candidateCommitted();
  assert(boundary_bad.observe(pose(1.0, 0.0), pose(1.0, 0.6), 1.0).state ==
         sm::State::ABORTED);
  assert(boundary_bad.evaluation().reason == "boundary_position_limit");

  sm::Selector growth_bad(config);
  growth_bad.candidateCommitted();
  growth_bad.observe(pose(1.0, 0.0), pose(1.0, 0.1), 1.0);
  assert(growth_bad.observe(pose(2.0, 1.0), pose(2.0, 1.7), 2.0).state ==
         sm::State::ABORTED);
  assert(growth_bad.evaluation().reason == "position_growth_limit");

  // A recovered generation may legitimately be far from a polluted
  // incumbent.  One fixed SE(3) alignment makes the switch boundary
  // continuous while retaining the raw correction magnitude for audit.
  sm::Config aligned_config = config;
  aligned_config.align_candidate_at_boundary = true;
  sm::Selector aligned_boundary(aligned_config);
  aligned_boundary.candidateCommitted();
  const sm::Evaluation &aligned_first = aligned_boundary.observe(
      pose(1.0, 100.0, 30.0), pose(1.0, 0.0, 0.0), 10.0);
  assert(aligned_first.state == sm::State::PROBATION);
  assert(aligned_first.boundary_alignment_applied);
  assert(std::abs(aligned_first.raw_boundary_position_m - 100.0) < 1e-12);
  assert(std::abs(aligned_first.raw_boundary_rotation_deg - 30.0) < 1e-12);
  assert(aligned_first.boundary_position_m < 1e-9);
  assert(aligned_first.boundary_rotation_deg < 1e-9);
  const double cos30 = std::cos(30.0 * 3.14159265358979323846 / 180.0);
  const double sin30 = std::sin(30.0 * 3.14159265358979323846 / 180.0);
  const sm::Evaluation &aligned_second = aligned_boundary.observe(
      pose(2.0, 100.0 + cos30, 30.0), pose(2.0, 1.0, 0.0), 10.0);
  assert(aligned_second.state == sm::State::PROBATION);
  sm::Pose full_incumbent = pose(2.0, 100.0 + cos30, 30.0);
  full_incumbent.y = sin30;
  sm::Pose full_candidate = pose(2.0, 1.0, 0.0);
  const sm::Pose aligned_pose = aligned_boundary.alignedCandidatePose(full_candidate);
  assert(std::abs(aligned_pose.x - full_incumbent.x) < 1e-9);
  assert(std::abs(aligned_pose.y - full_incumbent.y) < 1e-9);

  sm::Selector aligned_growth_bad(aligned_config);
  aligned_growth_bad.candidateCommitted();
  aligned_growth_bad.observe(pose(1.0, 100.0), pose(1.0, 0.0), 10.0);
  assert(aligned_growth_bad.observe(
      pose(2.0, 101.0), pose(2.0, 2.0), 10.0).state ==
      sm::State::ABORTED);
  assert(aligned_growth_bad.evaluation().reason == "position_growth_limit");

  sm::Selector frame_bad(config);
  frame_bad.candidateCommitted();
  sm::Pose wrong = pose(1.0, 0.0);
  wrong.frame_id = "wrong";
  assert(frame_bad.observe(pose(1.0, 0.0), wrong, 1.0).state ==
         sm::State::ABORTED);

  sm::Selector uncertainty_bad(config);
  uncertainty_bad.candidateCommitted();
  sm::Pose uncertain = pose(1.0, 0.0);
  uncertain.position_variance = 3.1;
  assert(uncertainty_bad.observe(pose(1.0, 0.0), uncertain, 1.0).state ==
         sm::State::ABORTED);
  assert(uncertainty_bad.evaluation().reason ==
         "position_uncertainty_ratio_limit");
  assert(std::abs(uncertainty_bad.evaluation().maximum_position_variance_ratio -
                  3.1) < 1e-12);
  assert(std::abs(uncertainty_bad.evaluation().position_variance_ratio_time_sec -
                  1.0) < 1e-12);
  assert(std::abs(
      uncertainty_bad.evaluation().candidate_position_variance_at_maximum_ratio -
      3.1) < 1e-12);
  assert(std::abs(
      uncertainty_bad.evaluation().incumbent_position_variance_at_maximum_ratio -
      1.0) < 1e-12);
  assert(uncertainty_bad.evaluation().uncertainty_samples == 1);
  assert(std::abs(uncertainty_bad.evaluation().mean_position_variance_ratio -
                  3.1) < 1e-12);

  sm::Selector rotation_uncertainty_bad(config);
  rotation_uncertainty_bad.candidateCommitted();
  sm::Pose rotation_uncertain = pose(2.0, 0.0);
  rotation_uncertain.rotation_variance = 2.3;
  assert(rotation_uncertainty_bad.observe(
      pose(2.0, 0.0), rotation_uncertain, 2.0).state == sm::State::ABORTED);
  assert(rotation_uncertainty_bad.evaluation().reason ==
         "rotation_uncertainty_ratio_limit");
  assert(std::abs(
      rotation_uncertainty_bad.evaluation().maximum_rotation_variance_ratio -
      2.3) < 1e-12);
  assert(std::abs(
      rotation_uncertainty_bad.evaluation().candidate_rotation_variance_at_maximum_ratio -
      2.3) < 1e-12);
  assert(std::abs(
      rotation_uncertainty_bad.evaluation().incumbent_rotation_variance_at_maximum_ratio -
      1.0) < 1e-12);

  sm::Config persistent_config = config;
  persistent_config.minimum_consecutive_uncertainty_exceedance_samples = 3;
  sm::Selector persistent_uncertainty(persistent_config);
  persistent_uncertainty.candidateCommitted();
  sm::Pose high_rotation = pose(1.0, 0.0);
  high_rotation.rotation_variance = 2.3;
  assert(persistent_uncertainty.observe(
      pose(1.0, 0.0), high_rotation, 10.0).state == sm::State::PROBATION);
  assert(persistent_uncertainty.evaluation().
      current_consecutive_rotation_uncertainty_exceedances == 1);
  assert(persistent_uncertainty.evaluation().
      maximum_consecutive_rotation_uncertainty_exceedances == 1);
  assert(persistent_uncertainty.evaluation().
      required_consecutive_uncertainty_exceedance_samples == 3);
  assert(persistent_uncertainty.observe(
      pose(2.0, 0.0), pose(2.0, 0.0), 10.0).state == sm::State::PROBATION);
  assert(persistent_uncertainty.evaluation().
      current_consecutive_rotation_uncertainty_exceedances == 0);
  high_rotation.time = 3.0;
  persistent_uncertainty.observe(pose(3.0, 0.0), high_rotation, 10.0);
  high_rotation.time = 4.0;
  persistent_uncertainty.observe(pose(4.0, 0.0), high_rotation, 10.0);
  assert(persistent_uncertainty.evaluation().state == sm::State::PROBATION);
  high_rotation.time = 5.0;
  assert(persistent_uncertainty.observe(
      pose(5.0, 0.0), high_rotation, 10.0).state == sm::State::ABORTED);
  assert(persistent_uncertainty.evaluation().reason ==
         "rotation_uncertainty_ratio_limit");
  assert(persistent_uncertainty.evaluation().
      rotation_uncertainty_exceedance_samples == 4);
  assert(persistent_uncertainty.evaluation().
      maximum_consecutive_rotation_uncertainty_exceedances == 3);

  sm::Selector commit_watermark(config);
  commit_watermark.candidateCommitted(10.0, 12.0);
  assert(commit_watermark.observe(
      pose(9.9, 0.0), pose(9.9, 1000.0), 12.0).state ==
      sm::State::PROBATION);
  assert(commit_watermark.evaluation().reason ==
         "waiting_for_post_commit_candidate_output");
  assert(commit_watermark.evaluation().precommit_candidate_samples_ignored == 1);
  assert(commit_watermark.evaluation().matched_samples == 0);
  assert(std::abs(commit_watermark.evaluation().candidate_commit_lag_sec - 2.0) <
         1e-12);
  const sm::Evaluation &post_commit = commit_watermark.observe(
      pose(10.1, 0.0), pose(10.1, 0.1), 12.0);
  assert(post_commit.state == sm::State::PROBATION);
  assert(std::abs(post_commit.boundary_position_m - 0.1) < 1e-12);
  assert(std::abs(post_commit.first_evaluated_candidate_time_sec - 10.1) <
         1e-12);
  assert(std::abs(post_commit.first_evaluated_incumbent_time_sec - 10.1) <
         1e-12);
  assert(std::abs(post_commit.first_evaluated_match_dt_sec) < 1e-12);
  assert(std::abs(
      post_commit.first_evaluated_candidate_minus_commit_sec - 0.1) < 1e-12);

  sm::Selector explicitly_rejected(config);
  explicitly_rejected.candidateCommitted();
  explicitly_rejected.candidateRejected("mapper_rejected");
  assert(explicitly_rejected.evaluation().state == sm::State::INCUMBENT);
  assert(explicitly_rejected.evaluation().reason ==
         "incumbent_retained_after_candidate_rejection");
  assert(explicitly_rejected.evaluation().candidate_rejected);
  assert(explicitly_rejected.evaluation().candidate_rejection_reason ==
         "mapper_rejected");
  explicitly_rejected.resetForNextCandidate();
  assert(explicitly_rejected.evaluation().state == sm::State::INCUMBENT);
  assert(explicitly_rejected.evaluation().reason == "incumbent_active");
  explicitly_rejected.candidateCommitted(20.0, 20.0);
  assert(explicitly_rejected.evaluation().state == sm::State::PROBATION);

  // Trial-52 regression: the mapper may reject during replay before an outer
  // commit/probation transition.  The active public generation stays incumbent
  // while the rejection remains independently observable.
  sm::Selector replay_rejected(config);
  replay_rejected.candidateRejected("healthy_tail_validation_failed");
  assert(replay_rejected.evaluation().state == sm::State::INCUMBENT);
  assert(replay_rejected.evaluation().candidate_rejected);
  assert(replay_rejected.evaluation().candidate_rejection_reason ==
         "healthy_tail_validation_failed");
  replay_rejected.resetForNextCandidate();
  assert(replay_rejected.evaluation().state == sm::State::INCUMBENT);
  assert(!replay_rejected.evaluation().candidate_rejected);

  bool invalid_thrown = false;
  try
  {
    sm::Config invalid;
    invalid.minimum_matched_samples = 0;
    sm::Selector selector(invalid);
  }
  catch (const std::invalid_argument &)
  {
    invalid_thrown = true;
  }
  assert(invalid_thrown);

  // The default-off bridge is held during probation and released only after
  // the public selector accepts; releasing during probation would recreate
  // the original 0.516 m jump at the actual switch.
  sm::Config bridge_config = config;
  bridge_config.probation_duration_sec = 3.0;
  bridge_config.correction_aware_position_bridge_enabled = true;
  sm::Selector bridge(bridge_config);
  bridge.candidateCommitted(10.0, 10.0);
  assert(bridge.observe(pose(10.0, 0.0), pose(10.0, 0.516), 10.0).state ==
         sm::State::PROBATION);
  assert(bridge.evaluation().position_bridge_applied);
  assert(std::abs(bridge.evaluation().boundary_position_m) < 1e-12);
  assert(std::abs(bridge.evaluation().position_bridge_peak_added_speed_mps -
                  0.09675) < 1e-10);
  assert(std::abs(bridge.alignedCandidatePose(pose(11.0, 1.516)).x - 1.0) <
         1e-12);
  bridge.observe(pose(11.0, 1.0), pose(11.0, 1.526), 11.0);
  bridge.observe(pose(12.0, 2.0), pose(12.0, 2.526), 12.0);
  bridge.observe(pose(13.0, 3.0), pose(13.0, 3.526), 13.0);
  assert(bridge.observe(pose(14.0, 4.0), pose(14.0, 4.526), 14.0).state ==
         sm::State::CANDIDATE);
  assert(!std::isfinite(bridge.evaluation().position_bridge_release_start_sec));
  bridge.startPositionBridgeRelease(14.0);
  assert(std::abs(bridge.evaluation().position_bridge_release_start_sec - 14.0) <
         1e-12);
  assert(std::abs(bridge.alignedCandidatePose(pose(14.0, 4.526)).x - 4.01) <
         1e-12);
  assert(std::abs(bridge.alignedCandidatePose(pose(18.0, 8.526)).x - 8.268) <
         1e-12);
  assert(std::abs(bridge.alignedCandidatePose(pose(22.0, 12.526)).x - 12.526) <
         1e-12);

  sm::Selector bridge_growth_bad(bridge_config);
  bridge_growth_bad.candidateCommitted(10.0, 10.0);
  bridge_growth_bad.observe(pose(10.0, 0.0), pose(10.0, 0.516), 10.0);
  assert(bridge_growth_bad.observe(
      pose(11.0, 1.0), pose(11.0, 2.316), 11.0).state == sm::State::ABORTED);
  assert(bridge_growth_bad.evaluation().reason == "position_growth_limit");

  sm::Selector bridge_covariance_bad(bridge_config);
  bridge_covariance_bad.candidateCommitted(10.0, 10.0);
  bridge_covariance_bad.observe(pose(10.0, 0.0), pose(10.0, 0.516), 10.0);
  sm::Pose high_covariance = pose(11.0, 1.526);
  high_covariance.position_variance = 4.0;
  assert(bridge_covariance_bad.observe(
      pose(11.0, 1.0), high_covariance, 11.0).state == sm::State::ABORTED);
  assert(bridge_covariance_bad.evaluation().reason ==
         "position_uncertainty_ratio_limit");

  sm::Selector bridge_handover_bad(bridge_config);
  bridge_handover_bad.candidateCommitted(10.0, 10.0);
  bridge_handover_bad.observe(pose(10.0, 0.0), pose(10.0, 0.516), 10.0);
  bridge_handover_bad.observe(pose(11.0, 1.0), pose(11.0, 1.636), 11.0);
  bridge_handover_bad.observe(pose(12.0, 2.0), pose(12.0, 2.636), 12.0);
  bridge_handover_bad.observe(pose(13.0, 3.0), pose(13.0, 3.636), 13.0);
  assert(bridge_handover_bad.observe(
      pose(14.0, 4.0), pose(14.0, 4.636), 14.0).state == sm::State::ABORTED);
  assert(bridge_handover_bad.evaluation().reason ==
         "handover_residual_position_limit");

  bool bridge_invalid_thrown = false;
  try
  {
    sm::Config conflicting = bridge_config;
    conflicting.align_candidate_at_boundary = true;
    sm::Selector selector(conflicting);
  }
  catch (const std::invalid_argument &)
  {
    bridge_invalid_thrown = true;
  }
  assert(bridge_invalid_thrown);

  invalid_thrown = false;
  try
  {
    sm::Config invalid;
    invalid.minimum_consecutive_uncertainty_exceedance_samples = 0;
    sm::Selector selector(invalid);
  }
  catch (const std::invalid_argument &)
  {
    invalid_thrown = true;
  }
  assert(invalid_thrown);

  std::cout << "shadow_generation_selector_test passed\n";
  return 0;
}
