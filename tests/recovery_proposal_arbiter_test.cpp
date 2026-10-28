#include "recovery_proposal_arbiter.h"

#include <cassert>
#include <iostream>

int main()
{
  fast_livo::recovery_scheduling::Config config;
  config.enabled = true;
  config.maximum_attempts_per_episode = 1;
  fast_livo::recovery_scheduling::RecoveryProposalArbiter arbiter(config);

  auto decision = arbiter.decide(1, 1);
  assert(decision.execute_recovery && !decision.duplicate_suppressed);
  decision = arbiter.decide(1, 2);
  assert(!decision.execute_recovery && decision.duplicate_suppressed);
  decision = arbiter.decide(1, 3);
  assert(!decision.execute_recovery && decision.duplicate_suppressed);
  decision = arbiter.decide(2, 1);
  assert(decision.execute_recovery && !decision.duplicate_suppressed);
  assert(arbiter.suppressedProposals() == 2);
  assert(arbiter.attemptsTotal() == 2);

  config.maximum_attempts_total = 3;
  fast_livo::recovery_scheduling::RecoveryProposalArbiter bounded(config);
  assert(bounded.decide(1, 1).execute_recovery);
  assert(bounded.decide(2, 1).execute_recovery);
  assert(bounded.decide(3, 1).execute_recovery);
  decision = bounded.decide(4, 1);
  assert(!decision.execute_recovery && decision.duplicate_suppressed);
  assert(decision.reason == "total_attempt_budget_exhausted");
  assert(bounded.attemptsTotal() == 3);

  config.maximum_attempts_total = 4;
  config.minimum_signal_escalation_ratio = 1.4;
  fast_livo::recovery_scheduling::RecoveryProposalArbiter escalating(config);
  assert(escalating.decide(1, 1, 2.0, 2.0, 2.0).execute_recovery);
  decision = escalating.decide(1, 2, 2.7, 2.1, 2.2);
  assert(!decision.execute_recovery);
  decision = escalating.decide(1, 3, 2.8, 2.1, 2.2);
  assert(decision.execute_recovery && decision.escalation_retry);
  assert(decision.reason == "signal_escalation_retry");
  decision = escalating.decide(1, 4, 3.8, 2.1, 2.2);
  assert(!decision.execute_recovery);
  decision = escalating.decide(1, 5, 4.0, 2.1, 2.2);
  assert(decision.execute_recovery && decision.escalation_retry);
  assert(escalating.attemptsTotal() == 3);
  assert(escalating.admittedAttemptsLifetimeTotal() == 3);
  assert(escalating.refundLastAttempt());
  assert(escalating.attemptsTotal() == 2);
  assert(escalating.admittedAttemptsLifetimeTotal() == 3);
  assert(!escalating.refundLastAttempt());
  decision = escalating.decide(2, 1, 1.0, 1.0, 1.0);
  assert(decision.execute_recovery);
  assert(escalating.attemptsTotal() == 3);

  config.maximum_attempts_total = 2;
  config.maximum_rejected_attempt_refunds_total = 2;
  config.minimum_signal_escalation_ratio = 0.0;
  fast_livo::recovery_scheduling::RecoveryProposalArbiter rejection_refunds(
      config);
  assert(rejection_refunds.decide(1, 1).execute_recovery);
  assert(rejection_refunds.refundLastRejectedAttempt());
  assert(rejection_refunds.attemptsTotal() == 0);
  assert(rejection_refunds.rejectedAttemptRefundsTotal() == 1);
  assert(rejection_refunds.decide(2, 1).execute_recovery);
  assert(rejection_refunds.refundLastRejectedAttempt());
  assert(rejection_refunds.attemptsTotal() == 0);
  assert(rejection_refunds.rejectedAttemptRefundsTotal() == 2);
  assert(rejection_refunds.decide(3, 1).execute_recovery);
  assert(!rejection_refunds.refundLastRejectedAttempt());
  assert(rejection_refunds.attemptsTotal() == 1);
  assert(rejection_refunds.decide(4, 1).execute_recovery);
  decision = rejection_refunds.decide(5, 1);
  assert(!decision.execute_recovery);
  assert(decision.reason == "total_attempt_budget_exhausted");
  assert(rejection_refunds.admittedAttemptsLifetimeTotal() == 4);
  assert(rejection_refunds.attemptsTotal() == 2);

  // Rejected attempts establish a cross-episode signal high-watermark.  The
  // already frozen 1.4x escalation ratio suppresses weaker periodic proposals
  // while still admitting a later fault with a materially stronger channel.
  config.maximum_attempts_total = 4;
  config.maximum_rejected_attempt_refunds_total = 4;
  config.minimum_signal_escalation_ratio = 1.4;
  fast_livo::recovery_scheduling::RecoveryProposalArbiter rejected_history(
      config);
  assert(rejected_history.decide(1, 1, 3.0, 2.0, 1.4).execute_recovery);
  assert(rejected_history.refundLastRejectedAttempt());
  decision = rejected_history.decide(2, 1, 2.2, 2.1, 1.8);
  assert(!decision.execute_recovery);
  assert(decision.reason ==
         "rejected_signal_high_watermark_not_exceeded");
  decision = rejected_history.decide(2, 2, 4.2, 1.0, 1.0);
  assert(decision.execute_recovery);
  assert(decision.rejected_history_escalation);

  // Relearning changes detector-normalized scores, so cross-episode residual
  // escalation may use the same-sequence raw statistic without changing the
  // frozen detector thresholds or the episode-local score budget.
  config.rejected_history_use_raw_residual = true;
  fast_livo::recovery_scheduling::RecoveryProposalArbiter raw_history(config);
  assert(raw_history.decide(1, 1, 3.0, 2.0, 1.0, 0.012).execute_recovery);
  assert(raw_history.refundLastRejectedAttempt());
  decision = raw_history.decide(2, 1, 1.1, 1.0, 1.0, 0.017);
  assert(decision.execute_recovery);
  assert(decision.rejected_history_escalation);
  assert(raw_history.refundLastRejectedAttempt());
  decision = raw_history.decide(3, 1, 5.0, 1.0, 1.0, 0.020);
  assert(!decision.execute_recovery);
  assert(decision.reason == "rejected_signal_high_watermark_not_exceeded");
  decision = raw_history.decide(3, 2, 1.0, 1.0, 1.0, 0.024);
  assert(decision.execute_recovery);

  config.maximum_rejected_attempt_refunds_total = 0;
  config.minimum_residual_score = 4.0;
  fast_livo::recovery_scheduling::RecoveryProposalArbiter residual_floor(
      config);
  decision = residual_floor.decide(1, 1, 3.999, 100.0, 100.0);
  assert(!decision.execute_recovery);
  assert(decision.reason == "residual_admission_floor_not_met");
  assert(residual_floor.attemptsTotal() == 0);
  decision = residual_floor.decide(1, 2, 4.0, 0.0, 0.0);
  assert(decision.execute_recovery);
  assert(residual_floor.attemptsTotal() == 1);

  config.enabled = false;
  fast_livo::recovery_scheduling::RecoveryProposalArbiter disabled(config);
  assert(disabled.decide(3, 1).execute_recovery);
  assert(disabled.decide(3, 2).execute_recovery);

  std::cout << "RECOVERY_PROPOSAL_ARBITER_TEST_OK" << std::endl;
  return 0;
}
