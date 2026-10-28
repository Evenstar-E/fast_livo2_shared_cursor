#ifndef RECOVERY_PROPOSAL_ARBITER_H_
#define RECOVERY_PROPOSAL_ARBITER_H_

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fast_livo
{
namespace recovery_scheduling
{

struct Config
{
  bool enabled = false;
  std::size_t maximum_attempts_per_episode = 1;
  // Zero preserves the historical unlimited cross-episode behavior.
  std::size_t maximum_attempts_total = 0;
  // Completed rejected replays may return their admission credit so that
  // natural precursors cannot consume the complete sequence budget before a
  // later fault.  This separate cap keeps physical replay work bounded by
  // maximum_attempts_total + maximum_rejected_attempt_refunds_total.  Zero
  // preserves the historical no-refund behavior for completed replays.
  std::size_t maximum_rejected_attempt_refunds_total = 0;
  // Optional P2 diagnostic admission floor. Zero disables it. The detector
  // still publishes every proposal; only expensive replay admission changes.
  double minimum_residual_score = 0.0;
  // A positive ratio permits a stronger proposal in the same merged episode
  // to consume another attempt. Zero keeps the original strict episode cap.
  double minimum_signal_escalation_ratio = 0.0;
  // Detector-normalized scores drift after a rejected attempt relearns its
  // baseline. Use the same-sequence raw residual only for the cross-episode
  // rejected-history comparison when the caller supplies it. Episode-local
  // budgets and the optional residual admission floor keep score semantics.
  bool rejected_history_use_raw_residual = false;

  void validate() const
  {
    if (enabled && maximum_attempts_per_episode == 0)
      throw std::invalid_argument(
          "recovery proposal episode budget must be positive");
    if (enabled && maximum_rejected_attempt_refunds_total > 0 &&
        maximum_attempts_total == 0)
      throw std::invalid_argument(
          "rejected-attempt refunds require a finite total attempt budget");
    if (!std::isfinite(minimum_residual_score) ||
        minimum_residual_score < 0.0)
      throw std::invalid_argument(
          "minimum residual admission score must be finite and non-negative");
    if (!std::isfinite(minimum_signal_escalation_ratio) ||
        minimum_signal_escalation_ratio < 0.0 ||
        (minimum_signal_escalation_ratio > 0.0 &&
         minimum_signal_escalation_ratio <= 1.0))
      throw std::invalid_argument(
          "proposal escalation ratio must be zero or greater than one");
  }
};

struct Decision
{
  bool execute_recovery = true;
  bool duplicate_suppressed = false;
  bool escalation_retry = false;
  bool rejected_history_escalation = false;
  std::size_t episode_id = 0;
  std::size_t proposal_index_in_episode = 0;
  std::size_t attempts_in_episode = 0;
  std::string reason = "arbiter_disabled";
};

// Converts the detector's complete proposal stream into a bounded recovery
// workload. The detector remains untouched and every proposal can still be
// logged; only expensive replay attempts are budgeted per merged episode.
class RecoveryProposalArbiter
{
public:
  explicit RecoveryProposalArbiter(const Config &config = Config())
      : config_(config)
  {
    config_.validate();
  }

  Decision decide(const std::size_t episode_id,
                  const std::size_t proposal_index_in_episode,
                  const double residual_score = 0.0,
                  const double match_score = 0.0,
                  const double innovation_score = 0.0,
                  const double raw_residual =
                      std::numeric_limits<double>::quiet_NaN())
  {
    last_attempt_refundable_ = false;
    if (episode_id == 0 || proposal_index_in_episode == 0)
      throw std::invalid_argument("invalid recovery proposal episode identity");
    Decision decision;
    decision.episode_id = episode_id;
    decision.proposal_index_in_episode = proposal_index_in_episode;
    if (!config_.enabled) return decision;

    const bool new_episode = episode_id != current_episode_id_;
    if (new_episode)
    {
      current_episode_id_ = episode_id;
      attempts_in_current_episode_ = 0;
      maximum_attempted_residual_score_ = 0.0;
      maximum_attempted_match_score_ = 0.0;
      maximum_attempted_innovation_score_ = 0.0;
    }
    if (config_.maximum_attempts_total > 0 &&
        attempts_total_ >= config_.maximum_attempts_total)
    {
      decision.execute_recovery = false;
      decision.duplicate_suppressed = true;
      decision.attempts_in_episode = attempts_in_current_episode_;
      decision.reason = "total_attempt_budget_exhausted";
      ++suppressed_proposals_;
      return decision;
    }
    if (config_.minimum_residual_score > 0.0 &&
        (!std::isfinite(residual_score) ||
         residual_score < config_.minimum_residual_score))
    {
      decision.execute_recovery = false;
      decision.duplicate_suppressed = true;
      decision.attempts_in_episode = attempts_in_current_episode_;
      decision.reason = "residual_admission_floor_not_met";
      ++suppressed_proposals_;
      return decision;
    }
    const double rejected_history_residual_signal =
        config_.rejected_history_use_raw_residual &&
        std::isfinite(raw_residual) && raw_residual >= 0.0 ?
            raw_residual : residual_score;
    const bool rejected_history_gate_enabled =
        attempts_in_current_episode_ == 0 && rejected_attempt_count_ > 0 &&
        config_.minimum_signal_escalation_ratio > 1.0;
    const auto escalated = [this](const double score,
                                  const double attempted_maximum)
    {
      return std::isfinite(score) && score > 0.0 &&
          attempted_maximum > 0.0 &&
          score >= attempted_maximum *
              config_.minimum_signal_escalation_ratio;
    };
    if (rejected_history_gate_enabled &&
        !escalated(rejected_history_residual_signal,
                   rejected_residual_high_watermark_) &&
        !escalated(match_score, rejected_match_high_watermark_) &&
        !escalated(innovation_score, rejected_innovation_high_watermark_))
    {
      decision.execute_recovery = false;
      decision.duplicate_suppressed = true;
      decision.attempts_in_episode = attempts_in_current_episode_;
      decision.reason = "rejected_signal_high_watermark_not_exceeded";
      ++suppressed_proposals_;
      return decision;
    }
    if (rejected_history_gate_enabled)
      decision.rejected_history_escalation = true;
    if (attempts_in_current_episode_ < config_.maximum_attempts_per_episode)
    {
      recordAttempt(residual_score, match_score, innovation_score,
                    rejected_history_residual_signal);
      decision.execute_recovery = true;
      decision.attempts_in_episode = attempts_in_current_episode_;
      decision.reason = "episode_attempt_budget_available";
      return decision;
    }

    const bool escalation_enabled =
        config_.minimum_signal_escalation_ratio > 1.0;
    if (escalation_enabled &&
        (escalated(residual_score, maximum_attempted_residual_score_) ||
         escalated(match_score, maximum_attempted_match_score_) ||
         escalated(innovation_score, maximum_attempted_innovation_score_)))
    {
      recordAttempt(residual_score, match_score, innovation_score,
                    rejected_history_residual_signal);
      decision.execute_recovery = true;
      decision.escalation_retry = true;
      decision.attempts_in_episode = attempts_in_current_episode_;
      decision.reason = "signal_escalation_retry";
      return decision;
    }

    decision.execute_recovery = false;
    decision.duplicate_suppressed = true;
    decision.attempts_in_episode = attempts_in_current_episode_;
    decision.reason = "episode_attempt_budget_exhausted";
    ++suppressed_proposals_;
    return decision;
  }

  std::size_t suppressedProposals() const { return suppressed_proposals_; }
  std::size_t attemptsTotal() const { return attempts_total_; }
  std::size_t admittedAttemptsLifetimeTotal() const
  {
    return admitted_attempts_lifetime_total_;
  }
  std::size_t rejectedAttemptRefundsTotal() const
  {
    return rejected_attempt_refunds_total_;
  }

  bool refundLastRejectedAttempt()
  {
    if (!last_attempt_refundable_) return false;
    rejected_residual_high_watermark_ = std::max(
        rejected_residual_high_watermark_,
        last_attempt_rejected_history_residual_signal_);
    rejected_match_high_watermark_ = std::max(
        rejected_match_high_watermark_, last_attempt_match_score_);
    rejected_innovation_high_watermark_ = std::max(
        rejected_innovation_high_watermark_, last_attempt_innovation_score_);
    ++rejected_attempt_count_;
    if (config_.maximum_rejected_attempt_refunds_total == 0 ||
        rejected_attempt_refunds_total_ >=
            config_.maximum_rejected_attempt_refunds_total)
      return false;
    if (!refundLastAttempt()) return false;
    ++rejected_attempt_refunds_total_;
    return true;
  }

  bool refundLastAttempt()
  {
    if (!last_attempt_refundable_ || attempts_total_ == 0) return false;
    attempts_in_current_episode_ = attempts_before_last_attempt_;
    --attempts_total_;
    maximum_attempted_residual_score_ =
        residual_maximum_before_last_attempt_;
    maximum_attempted_match_score_ = match_maximum_before_last_attempt_;
    maximum_attempted_innovation_score_ =
        innovation_maximum_before_last_attempt_;
    last_attempt_refundable_ = false;
    return true;
  }

private:
  void recordAttempt(const double residual_score, const double match_score,
                     const double innovation_score,
                     const double rejected_history_residual_signal)
  {
    attempts_before_last_attempt_ = attempts_in_current_episode_;
    residual_maximum_before_last_attempt_ =
        maximum_attempted_residual_score_;
    match_maximum_before_last_attempt_ = maximum_attempted_match_score_;
    innovation_maximum_before_last_attempt_ =
        maximum_attempted_innovation_score_;
    ++attempts_in_current_episode_;
    ++attempts_total_;
    ++admitted_attempts_lifetime_total_;
    if (std::isfinite(residual_score))
      maximum_attempted_residual_score_ = std::max(
          maximum_attempted_residual_score_, residual_score);
    if (std::isfinite(match_score))
      maximum_attempted_match_score_ = std::max(
          maximum_attempted_match_score_, match_score);
    if (std::isfinite(innovation_score))
      maximum_attempted_innovation_score_ = std::max(
          maximum_attempted_innovation_score_, innovation_score);
    last_attempt_residual_score_ =
        std::isfinite(residual_score) ? residual_score : 0.0;
    last_attempt_rejected_history_residual_signal_ =
        std::isfinite(rejected_history_residual_signal) ?
            rejected_history_residual_signal : 0.0;
    last_attempt_match_score_ =
        std::isfinite(match_score) ? match_score : 0.0;
    last_attempt_innovation_score_ =
        std::isfinite(innovation_score) ? innovation_score : 0.0;
    last_attempt_refundable_ = true;
  }

  Config config_;
  std::size_t current_episode_id_ = 0;
  std::size_t attempts_in_current_episode_ = 0;
  std::size_t attempts_total_ = 0;
  std::size_t admitted_attempts_lifetime_total_ = 0;
  std::size_t rejected_attempt_refunds_total_ = 0;
  double maximum_attempted_residual_score_ = 0.0;
  double maximum_attempted_match_score_ = 0.0;
  double maximum_attempted_innovation_score_ = 0.0;
  bool last_attempt_refundable_ = false;
  std::size_t attempts_before_last_attempt_ = 0;
  double residual_maximum_before_last_attempt_ = 0.0;
  double match_maximum_before_last_attempt_ = 0.0;
  double innovation_maximum_before_last_attempt_ = 0.0;
  double last_attempt_residual_score_ = 0.0;
  double last_attempt_rejected_history_residual_signal_ = 0.0;
  double last_attempt_match_score_ = 0.0;
  double last_attempt_innovation_score_ = 0.0;
  std::size_t rejected_attempt_count_ = 0;
  double rejected_residual_high_watermark_ = 0.0;
  double rejected_match_high_watermark_ = 0.0;
  double rejected_innovation_high_watermark_ = 0.0;
  std::size_t suppressed_proposals_ = 0;
};

} // namespace recovery_scheduling
} // namespace fast_livo

#endif // RECOVERY_PROPOSAL_ARBITER_H_
