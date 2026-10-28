#ifndef HEALTHY_TAIL_VALIDATOR_H_
#define HEALTHY_TAIL_VALIDATOR_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fast_livo
{
namespace recovery_validation
{

struct Config
{
  bool enabled = false;
  std::size_t consecutive_healthy = 20;
  double minimum_healthy_duration_sec = 1.5;
  double timeout_sec = 12.0;
  bool sequential_evidence_enabled = false;
  std::size_t minimum_sequential_observations = 20;
  double confidence_z = 1.645;
  double minimum_residual_pass_rate_lower_bound = 0.60;
  double minimum_match_pass_rate_lower_bound = 0.60;
  double minimum_joint_pass_rate_lower_bound = 0.55;
  bool candidate_summary_required = false;
  double maximum_reentry_position_correction_m = 0.50;
  double maximum_reentry_rotation_correction_deg = 5.0;
  double minimum_map_coverage_ratio = 1.0;
  double minimum_map_support_ratio = 0.75;
  double maximum_boundary_position_divergence_m = 1.0;
  double maximum_boundary_rotation_divergence_deg = 5.0;
  bool use_residual_check = true;
  bool use_match_ratio_check = true;
  bool use_reentry_position_check = true;
  bool use_reentry_rotation_check = true;
  bool use_map_coverage_check = true;
  bool use_map_support_check = true;
  bool use_boundary_position_check = true;
  bool use_boundary_rotation_check = true;

  void validate() const
  {
    if (!enabled) return;
    if (consecutive_healthy == 0 || minimum_healthy_duration_sec < 0.0 ||
        timeout_sec <= minimum_healthy_duration_sec ||
        maximum_reentry_position_correction_m <= 0.0 ||
        maximum_reentry_rotation_correction_deg <= 0.0 ||
        minimum_map_coverage_ratio <= 0.0 || minimum_map_support_ratio <= 0.0 ||
        maximum_boundary_position_divergence_m <= 0.0 ||
        maximum_boundary_rotation_divergence_deg <= 0.0)
      throw std::invalid_argument("invalid healthy-tail validator configuration");
    if (sequential_evidence_enabled &&
        (minimum_sequential_observations == 0 || confidence_z <= 0.0 ||
         minimum_residual_pass_rate_lower_bound < 0.0 ||
         minimum_residual_pass_rate_lower_bound > 1.0 ||
         minimum_match_pass_rate_lower_bound < 0.0 ||
         minimum_match_pass_rate_lower_bound > 1.0 ||
         minimum_joint_pass_rate_lower_bound < 0.0 ||
         minimum_joint_pass_rate_lower_bound > 1.0))
      throw std::invalid_argument(
          "invalid sequential healthy-tail confidence configuration");
  }
};

struct Decision
{
  bool healthy_sample = false;
  bool residual_pass = false;
  bool match_ratio_pass = false;
  bool approved = false;
  bool timed_out = false;
  std::size_t consecutive_healthy = 0;
  double healthy_tail_start_sec = -1.0;
  double healthy_run_duration_sec = 0.0;
  double residual_margin = std::numeric_limits<double>::quiet_NaN();
  double match_ratio_margin = std::numeric_limits<double>::quiet_NaN();
  double residual_pass_rate_lower_bound = 0.0;
  double match_pass_rate_lower_bound = 0.0;
  double joint_pass_rate_lower_bound = 0.0;
  bool sequential_evidence_pass = false;
};

struct TailDiagnostics
{
  std::size_t observation_count = 0;
  std::size_t residual_pass_samples = 0;
  std::size_t match_ratio_pass_samples = 0;
  std::size_t jointly_healthy_samples = 0;
  std::size_t longest_consecutive_healthy = 0;
  double longest_healthy_duration_sec = 0.0;
  double minimum_residual_margin = std::numeric_limits<double>::quiet_NaN();
  double minimum_match_ratio_margin = std::numeric_limits<double>::quiet_NaN();
  double residual_limit = std::numeric_limits<double>::quiet_NaN();
  double match_ratio_limit = std::numeric_limits<double>::quiet_NaN();
  bool residual_check_enabled = true;
  bool match_ratio_check_enabled = true;
  bool sequential_evidence_enabled = false;
  bool sequential_evidence_pass = false;
  double observation_duration_sec = 0.0;
  double residual_pass_rate = 0.0;
  double match_pass_rate = 0.0;
  double joint_pass_rate = 0.0;
  double residual_pass_rate_lower_bound = 0.0;
  double match_pass_rate_lower_bound = 0.0;
  double joint_pass_rate_lower_bound = 0.0;
  bool approved = false;
};

struct CandidateSummaryDiagnostics
{
  bool set = false;
  bool healthy = false;
  double reentry_position_correction_m = std::numeric_limits<double>::quiet_NaN();
  double reentry_rotation_correction_deg = std::numeric_limits<double>::quiet_NaN();
  double map_coverage_ratio = std::numeric_limits<double>::quiet_NaN();
  double map_support_ratio = std::numeric_limits<double>::quiet_NaN();
  double boundary_position_divergence_m = std::numeric_limits<double>::quiet_NaN();
  double boundary_rotation_divergence_deg = std::numeric_limits<double>::quiet_NaN();
  double reentry_position_margin_m = std::numeric_limits<double>::quiet_NaN();
  double reentry_rotation_margin_deg = std::numeric_limits<double>::quiet_NaN();
  double map_coverage_margin = std::numeric_limits<double>::quiet_NaN();
  double map_support_margin = std::numeric_limits<double>::quiet_NaN();
  double boundary_position_margin_m = std::numeric_limits<double>::quiet_NaN();
  double boundary_rotation_margin_deg = std::numeric_limits<double>::quiet_NaN();
  bool reentry_position_pass = false;
  bool reentry_rotation_pass = false;
  bool map_coverage_pass = false;
  bool map_support_pass = false;
  bool boundary_position_pass = false;
  bool boundary_rotation_pass = false;
};

class HealthyTailValidator
{
public:
  explicit HealthyTailValidator(const Config &config = Config()) : config_(config)
  {
    config_.validate();
  }

  void arm(double trigger_time_sec, double residual_limit, double match_ratio_limit)
  {
    if (!std::isfinite(trigger_time_sec) || trigger_time_sec < 0.0 ||
        !std::isfinite(residual_limit) || residual_limit <= 0.0 ||
        !std::isfinite(match_ratio_limit) || match_ratio_limit < 0.0 ||
        match_ratio_limit > 1.0)
      throw std::invalid_argument("invalid healthy-tail validator limits");
    trigger_time_sec_ = trigger_time_sec;
    residual_limit_ = residual_limit;
    match_ratio_limit_ = match_ratio_limit;
    healthy_run_start_sec_ = -1.0;
    observation_start_sec_ = -1.0;
    last_observation_sec_ = -1.0;
    consecutive_healthy_ = 0;
    approved_ = false;
    match_ratio_sum_ = 0.0;
    observation_count_ = 0;
    residual_pass_samples_ = 0;
    match_ratio_pass_samples_ = 0;
    jointly_healthy_samples_ = 0;
    longest_consecutive_healthy_ = 0;
    longest_healthy_duration_sec_ = 0.0;
    minimum_residual_margin_ = std::numeric_limits<double>::quiet_NaN();
    minimum_match_ratio_margin_ = std::numeric_limits<double>::quiet_NaN();
    candidate_summary_set_ = false;
    candidate_summary_healthy_ = false;
    candidate_summary_diagnostics_ = CandidateSummaryDiagnostics();
    armed_ = true;
  }

  Decision observe(double time_sec, double mean_abs_residual, double match_ratio)
  {
    if (!armed_ || !config_.enabled) return Decision();
    if (!std::isfinite(time_sec) || !std::isfinite(mean_abs_residual) ||
        !std::isfinite(match_ratio) || time_sec < trigger_time_sec_ ||
        mean_abs_residual < 0.0 || match_ratio < 0.0 || match_ratio > 1.0)
      throw std::invalid_argument("invalid healthy-tail observation");

    Decision decision;
    if (observation_start_sec_ < 0.0) observation_start_sec_ = time_sec;
    last_observation_sec_ = time_sec;
    match_ratio_sum_ += match_ratio;
    ++observation_count_;
    decision.residual_margin = residual_limit_ - mean_abs_residual;
    decision.match_ratio_margin = match_ratio - match_ratio_limit_;
    decision.residual_pass = !config_.use_residual_check ||
                             decision.residual_margin >= 0.0;
    decision.match_ratio_pass = !config_.use_match_ratio_check ||
                                decision.match_ratio_margin >= 0.0;
    if (decision.residual_margin >= 0.0) ++residual_pass_samples_;
    if (decision.match_ratio_margin >= 0.0) ++match_ratio_pass_samples_;
    minimum_residual_margin_ = minimumMargin(
        minimum_residual_margin_, decision.residual_margin);
    minimum_match_ratio_margin_ = minimumMargin(
        minimum_match_ratio_margin_, decision.match_ratio_margin);
    decision.healthy_sample = decision.residual_pass && decision.match_ratio_pass;
    if (decision.healthy_sample) ++jointly_healthy_samples_;
    decision.residual_pass_rate_lower_bound = config_.use_residual_check ?
        wilsonLowerBound(residual_pass_samples_, observation_count_,
                         config_.confidence_z) : 1.0;
    decision.match_pass_rate_lower_bound = config_.use_match_ratio_check ?
        wilsonLowerBound(match_ratio_pass_samples_, observation_count_,
                         config_.confidence_z) : 1.0;
    decision.joint_pass_rate_lower_bound = wilsonLowerBound(
        jointly_healthy_samples_, observation_count_, config_.confidence_z);
    const double observation_duration = time_sec - observation_start_sec_;
    decision.sequential_evidence_pass =
        config_.sequential_evidence_enabled &&
        observation_count_ >= config_.minimum_sequential_observations &&
        observation_duration >= config_.minimum_healthy_duration_sec &&
        decision.residual_pass_rate_lower_bound >=
            config_.minimum_residual_pass_rate_lower_bound &&
        decision.match_pass_rate_lower_bound >=
            config_.minimum_match_pass_rate_lower_bound &&
        decision.joint_pass_rate_lower_bound >=
            config_.minimum_joint_pass_rate_lower_bound;
    if (!approved_ && !config_.sequential_evidence_enabled &&
        decision.healthy_sample)
    {
      if (consecutive_healthy_ == 0) healthy_run_start_sec_ = time_sec;
      ++consecutive_healthy_;
      const double duration = time_sec - healthy_run_start_sec_;
      longest_consecutive_healthy_ = std::max(
          longest_consecutive_healthy_, consecutive_healthy_);
      longest_healthy_duration_sec_ = std::max(
          longest_healthy_duration_sec_, duration);
      if (consecutive_healthy_ >= config_.consecutive_healthy &&
          duration >= config_.minimum_healthy_duration_sec)
        approved_ = true;
    }
    else if (!approved_ && !config_.sequential_evidence_enabled)
    {
      consecutive_healthy_ = 0;
      healthy_run_start_sec_ = -1.0;
    }
    // Sequential approval is a property of the complete accumulated tail,
    // not a latched prefix event. Later evidence must be able to revoke an
    // early pass before the final commit decision is made.
    if (config_.sequential_evidence_enabled)
      approved_ = decision.sequential_evidence_pass;

    decision.approved = approved_;
    decision.timed_out = !approved_ && time_sec - trigger_time_sec_ >= config_.timeout_sec;
    decision.consecutive_healthy = consecutive_healthy_;
    decision.healthy_tail_start_sec = config_.sequential_evidence_enabled ?
        observation_start_sec_ : healthy_run_start_sec_;
    decision.healthy_run_duration_sec = decision.healthy_tail_start_sec >= 0.0 ?
        time_sec - decision.healthy_tail_start_sec : 0.0;
    return decision;
  }

  bool armed() const { return armed_; }
  bool approved() const { return approved_; }
  bool localHealthApproved() const { return approved_; }
  void setCandidateSummary(double reentry_position_correction_m,
                           double reentry_rotation_correction_deg,
                           double map_coverage_ratio,
                           double map_support_ratio,
                           double boundary_position_divergence_m,
                           double boundary_rotation_divergence_deg)
  {
    if (!std::isfinite(reentry_position_correction_m) ||
        !std::isfinite(reentry_rotation_correction_deg) ||
        !std::isfinite(map_coverage_ratio) || !std::isfinite(map_support_ratio) ||
        !std::isfinite(boundary_position_divergence_m) ||
        !std::isfinite(boundary_rotation_divergence_deg) ||
        reentry_position_correction_m < 0.0 ||
        reentry_rotation_correction_deg < 0.0 || map_coverage_ratio < 0.0 ||
        map_support_ratio < 0.0 || boundary_position_divergence_m < 0.0 ||
        boundary_rotation_divergence_deg < 0.0)
      throw std::invalid_argument("invalid candidate summary");
    candidate_summary_diagnostics_.set = true;
    candidate_summary_diagnostics_.reentry_position_correction_m =
        reentry_position_correction_m;
    candidate_summary_diagnostics_.reentry_rotation_correction_deg =
        reentry_rotation_correction_deg;
    candidate_summary_diagnostics_.map_coverage_ratio = map_coverage_ratio;
    candidate_summary_diagnostics_.map_support_ratio = map_support_ratio;
    candidate_summary_diagnostics_.boundary_position_divergence_m =
        boundary_position_divergence_m;
    candidate_summary_diagnostics_.boundary_rotation_divergence_deg =
        boundary_rotation_divergence_deg;
    candidate_summary_diagnostics_.reentry_position_margin_m =
        config_.maximum_reentry_position_correction_m -
            reentry_position_correction_m;
    candidate_summary_diagnostics_.reentry_rotation_margin_deg =
        config_.maximum_reentry_rotation_correction_deg -
            reentry_rotation_correction_deg;
    candidate_summary_diagnostics_.map_coverage_margin =
        map_coverage_ratio - config_.minimum_map_coverage_ratio;
    candidate_summary_diagnostics_.map_support_margin =
        map_support_ratio - config_.minimum_map_support_ratio;
    candidate_summary_diagnostics_.boundary_position_margin_m =
        config_.maximum_boundary_position_divergence_m -
            boundary_position_divergence_m;
    candidate_summary_diagnostics_.boundary_rotation_margin_deg =
        config_.maximum_boundary_rotation_divergence_deg -
            boundary_rotation_divergence_deg;
    candidate_summary_diagnostics_.reentry_position_pass =
        !config_.use_reentry_position_check ||
        candidate_summary_diagnostics_.reentry_position_margin_m >= 0.0;
    candidate_summary_diagnostics_.reentry_rotation_pass =
        !config_.use_reentry_rotation_check ||
        candidate_summary_diagnostics_.reentry_rotation_margin_deg >= 0.0;
    candidate_summary_diagnostics_.map_coverage_pass =
        !config_.use_map_coverage_check ||
        candidate_summary_diagnostics_.map_coverage_margin >= 0.0;
    candidate_summary_diagnostics_.map_support_pass =
        !config_.use_map_support_check ||
        candidate_summary_diagnostics_.map_support_margin >= 0.0;
    candidate_summary_diagnostics_.boundary_position_pass =
        !config_.use_boundary_position_check ||
        candidate_summary_diagnostics_.boundary_position_margin_m >= 0.0;
    candidate_summary_diagnostics_.boundary_rotation_pass =
        !config_.use_boundary_rotation_check ||
        candidate_summary_diagnostics_.boundary_rotation_margin_deg >= 0.0;
    candidate_summary_healthy_ =
        candidate_summary_diagnostics_.reentry_position_pass &&
        candidate_summary_diagnostics_.reentry_rotation_pass &&
        candidate_summary_diagnostics_.map_coverage_pass &&
        candidate_summary_diagnostics_.map_support_pass &&
        candidate_summary_diagnostics_.boundary_position_pass &&
        candidate_summary_diagnostics_.boundary_rotation_pass;
    candidate_summary_diagnostics_.healthy = candidate_summary_healthy_;
    candidate_summary_set_ = true;
  }
  bool commitApproved() const
  {
    return approved_ && (!config_.candidate_summary_required ||
                         (candidate_summary_set_ && candidate_summary_healthy_));
  }
  bool candidateSummaryHealthy() const
  {
    return candidate_summary_set_ && candidate_summary_healthy_;
  }
  bool mapAcceptable() const
  {
    return candidate_summary_set_ &&
        candidate_summary_diagnostics_.map_coverage_pass &&
        candidate_summary_diagnostics_.map_support_pass;
  }
  bool geometricContinuityAcceptable() const
  {
    return candidate_summary_set_ &&
        candidate_summary_diagnostics_.reentry_position_pass &&
        candidate_summary_diagnostics_.reentry_rotation_pass &&
        candidate_summary_diagnostics_.boundary_position_pass &&
        candidate_summary_diagnostics_.boundary_rotation_pass;
  }
  double healthyTailStartTime() const
  {
    return config_.sequential_evidence_enabled ? observation_start_sec_ :
                                                healthy_run_start_sec_;
  }
  TailDiagnostics tailDiagnostics() const
  {
    TailDiagnostics diagnostics;
    diagnostics.observation_count = observation_count_;
    diagnostics.residual_pass_samples = residual_pass_samples_;
    diagnostics.match_ratio_pass_samples = match_ratio_pass_samples_;
    diagnostics.jointly_healthy_samples = jointly_healthy_samples_;
    diagnostics.longest_consecutive_healthy = longest_consecutive_healthy_;
    diagnostics.longest_healthy_duration_sec = longest_healthy_duration_sec_;
    diagnostics.minimum_residual_margin = minimum_residual_margin_;
    diagnostics.minimum_match_ratio_margin = minimum_match_ratio_margin_;
    diagnostics.residual_limit = residual_limit_;
    diagnostics.match_ratio_limit = match_ratio_limit_;
    diagnostics.residual_check_enabled = config_.use_residual_check;
    diagnostics.match_ratio_check_enabled = config_.use_match_ratio_check;
    diagnostics.sequential_evidence_enabled = config_.sequential_evidence_enabled;
    diagnostics.observation_duration_sec = observation_start_sec_ >= 0.0 &&
        last_observation_sec_ >= observation_start_sec_ ?
        last_observation_sec_ - observation_start_sec_ : 0.0;
    diagnostics.residual_pass_rate = passRate(
        residual_pass_samples_, observation_count_);
    diagnostics.match_pass_rate = passRate(
        match_ratio_pass_samples_, observation_count_);
    diagnostics.joint_pass_rate = passRate(
        jointly_healthy_samples_, observation_count_);
    diagnostics.residual_pass_rate_lower_bound = config_.use_residual_check ?
        wilsonLowerBound(residual_pass_samples_, observation_count_,
                         config_.confidence_z) : 1.0;
    diagnostics.match_pass_rate_lower_bound = config_.use_match_ratio_check ?
        wilsonLowerBound(match_ratio_pass_samples_, observation_count_,
                         config_.confidence_z) : 1.0;
    diagnostics.joint_pass_rate_lower_bound = wilsonLowerBound(
        jointly_healthy_samples_, observation_count_, config_.confidence_z);
    diagnostics.sequential_evidence_pass =
        config_.sequential_evidence_enabled &&
        observation_count_ >= config_.minimum_sequential_observations &&
        diagnostics.observation_duration_sec >= config_.minimum_healthy_duration_sec &&
        diagnostics.residual_pass_rate_lower_bound >=
            config_.minimum_residual_pass_rate_lower_bound &&
        diagnostics.match_pass_rate_lower_bound >=
            config_.minimum_match_pass_rate_lower_bound &&
        diagnostics.joint_pass_rate_lower_bound >=
            config_.minimum_joint_pass_rate_lower_bound;
    diagnostics.approved = approved_;
    return diagnostics;
  }
  const CandidateSummaryDiagnostics &candidateSummaryDiagnostics() const
  {
    return candidate_summary_diagnostics_;
  }
  std::string failedChecks() const
  {
    std::string result;
    if (!approved_) appendCheck(result, config_.sequential_evidence_enabled ?
        "healthy_tail_sequential_confidence" : "healthy_tail_sequence");
    if (config_.candidate_summary_required && !candidate_summary_set_)
      appendCheck(result, "candidate_summary_missing");
    if (candidate_summary_set_)
    {
      if (!candidate_summary_diagnostics_.reentry_position_pass)
        appendCheck(result, "reentry_position");
      if (!candidate_summary_diagnostics_.reentry_rotation_pass)
        appendCheck(result, "reentry_rotation");
      if (!candidate_summary_diagnostics_.map_coverage_pass)
        appendCheck(result, "map_coverage");
      if (!candidate_summary_diagnostics_.map_support_pass)
        appendCheck(result, "map_support");
      if (!candidate_summary_diagnostics_.boundary_position_pass)
        appendCheck(result, "boundary_position");
      if (!candidate_summary_diagnostics_.boundary_rotation_pass)
        appendCheck(result, "boundary_rotation");
    }
    return result.empty() ? "none" : result;
  }
  std::string diagnosticFields() const
  {
    const TailDiagnostics tail = tailDiagnostics();
    const CandidateSummaryDiagnostics &summary = candidate_summary_diagnostics_;
    const double effective_tail_duration_sec =
        config_.sequential_evidence_enabled ? tail.observation_duration_sec :
                                              tail.longest_healthy_duration_sec;
    std::ostringstream stream;
    stream << std::setprecision(17)
        << "healthy_tail_failed_checks=" << failedChecks()
        << ";healthy_tail_approved=" << static_cast<int>(tail.approved)
        << ";healthy_tail_observation_count=" << tail.observation_count
        << ";healthy_tail_residual_pass_samples=" << tail.residual_pass_samples
        << ";healthy_tail_match_pass_samples=" << tail.match_ratio_pass_samples
        << ";healthy_tail_joint_pass_samples=" << tail.jointly_healthy_samples
        << ";healthy_tail_longest_consecutive="
        << tail.longest_consecutive_healthy
        << ";healthy_tail_required_consecutive=" << config_.consecutive_healthy
        << ";healthy_tail_consecutive_margin_samples="
        << static_cast<long long>(tail.longest_consecutive_healthy) -
               static_cast<long long>(config_.consecutive_healthy)
        << ";healthy_tail_longest_duration_sec="
        << tail.longest_healthy_duration_sec
        << ";healthy_tail_required_duration_sec="
        << config_.minimum_healthy_duration_sec
        << ";healthy_tail_duration_margin_sec="
        << effective_tail_duration_sec - config_.minimum_healthy_duration_sec
        << ";healthy_tail_residual_limit=" << tail.residual_limit
        << ";healthy_tail_minimum_residual_margin="
        << tail.minimum_residual_margin
        << ";healthy_tail_match_ratio_limit=" << tail.match_ratio_limit
        << ";healthy_tail_minimum_match_margin="
        << tail.minimum_match_ratio_margin
        << ";healthy_tail_residual_check_enabled="
        << static_cast<int>(tail.residual_check_enabled)
        << ";healthy_tail_match_check_enabled="
        << static_cast<int>(tail.match_ratio_check_enabled)
        << ";healthy_tail_sequential_evidence_enabled="
        << static_cast<int>(tail.sequential_evidence_enabled)
        << ";healthy_tail_sequential_evidence_pass="
        << static_cast<int>(tail.sequential_evidence_pass)
        << ";healthy_tail_observation_duration_sec="
        << tail.observation_duration_sec
        << ";healthy_tail_minimum_sequential_observations="
        << config_.minimum_sequential_observations
        << ";healthy_tail_sequential_observation_margin_samples="
        << static_cast<long long>(tail.observation_count) -
               static_cast<long long>(config_.minimum_sequential_observations)
        << ";healthy_tail_confidence_z=" << config_.confidence_z
        << ";healthy_tail_residual_pass_rate=" << tail.residual_pass_rate
        << ";healthy_tail_residual_pass_rate_lower_bound="
        << tail.residual_pass_rate_lower_bound
        << ";healthy_tail_residual_pass_rate_required_lower_bound="
        << config_.minimum_residual_pass_rate_lower_bound
        << ";healthy_tail_residual_pass_rate_margin="
        << tail.residual_pass_rate_lower_bound -
               config_.minimum_residual_pass_rate_lower_bound
        << ";healthy_tail_match_pass_rate=" << tail.match_pass_rate
        << ";healthy_tail_match_pass_rate_lower_bound="
        << tail.match_pass_rate_lower_bound
        << ";healthy_tail_match_pass_rate_required_lower_bound="
        << config_.minimum_match_pass_rate_lower_bound
        << ";healthy_tail_match_pass_rate_margin="
        << tail.match_pass_rate_lower_bound -
               config_.minimum_match_pass_rate_lower_bound
        << ";healthy_tail_joint_pass_rate=" << tail.joint_pass_rate
        << ";healthy_tail_joint_pass_rate_lower_bound="
        << tail.joint_pass_rate_lower_bound
        << ";healthy_tail_joint_pass_rate_required_lower_bound="
        << config_.minimum_joint_pass_rate_lower_bound
        << ";healthy_tail_joint_pass_rate_margin="
        << tail.joint_pass_rate_lower_bound -
               config_.minimum_joint_pass_rate_lower_bound
        << ";candidate_summary_set=" << static_cast<int>(summary.set)
        << ";candidate_summary_healthy=" << static_cast<int>(summary.healthy)
        << summaryFields(summary);
    return stream.str();
  }
  double meanObservedMatchRatio() const
  {
    return observation_count_ > 0 ?
        match_ratio_sum_ / static_cast<double>(observation_count_) : 0.0;
  }

private:
  static double passRate(const std::size_t successes, const std::size_t trials)
  {
    return trials > 0 ? static_cast<double>(successes) /
        static_cast<double>(trials) : 0.0;
  }

  static double wilsonLowerBound(const std::size_t successes,
                                 const std::size_t trials,
                                 const double z)
  {
    if (trials == 0) return 0.0;
    const double n = static_cast<double>(trials);
    const double p = static_cast<double>(successes) / n;
    const double z2 = z * z;
    const double center = p + z2 / (2.0 * n);
    const double radius = z * std::sqrt(
        (p * (1.0 - p) + z2 / (4.0 * n)) / n);
    return std::max(0.0, (center - radius) / (1.0 + z2 / n));
  }

  static double minimumMargin(const double current, const double value)
  {
    return std::isfinite(current) ? std::min(current, value) : value;
  }

  static void appendCheck(std::string &checks, const std::string &check)
  {
    if (!checks.empty()) checks += ',';
    checks += check;
  }

  std::string summaryFields(const CandidateSummaryDiagnostics &summary) const
  {
    std::ostringstream stream;
    stream << std::setprecision(17)
        << ";reentry_position_observed_m=" << summary.reentry_position_correction_m
        << ";reentry_position_limit_m="
        << config_.maximum_reentry_position_correction_m
        << ";reentry_position_margin_m=" << summary.reentry_position_margin_m
        << ";reentry_position_pass=" << static_cast<int>(summary.reentry_position_pass)
        << ";reentry_rotation_observed_deg=" << summary.reentry_rotation_correction_deg
        << ";reentry_rotation_limit_deg="
        << config_.maximum_reentry_rotation_correction_deg
        << ";reentry_rotation_margin_deg=" << summary.reentry_rotation_margin_deg
        << ";reentry_rotation_pass=" << static_cast<int>(summary.reentry_rotation_pass)
        << ";map_coverage_observed_ratio=" << summary.map_coverage_ratio
        << ";map_coverage_limit_ratio=" << config_.minimum_map_coverage_ratio
        << ";map_coverage_margin=" << summary.map_coverage_margin
        << ";map_coverage_pass=" << static_cast<int>(summary.map_coverage_pass)
        << ";map_support_observed_ratio=" << summary.map_support_ratio
        << ";map_support_limit_ratio=" << config_.minimum_map_support_ratio
        << ";map_support_margin=" << summary.map_support_margin
        << ";map_support_pass=" << static_cast<int>(summary.map_support_pass)
        << ";boundary_position_observed_m="
        << summary.boundary_position_divergence_m
        << ";boundary_position_limit_m="
        << config_.maximum_boundary_position_divergence_m
        << ";boundary_position_margin_m=" << summary.boundary_position_margin_m
        << ";boundary_position_pass="
        << static_cast<int>(summary.boundary_position_pass)
        << ";boundary_rotation_observed_deg="
        << summary.boundary_rotation_divergence_deg
        << ";boundary_rotation_limit_deg="
        << config_.maximum_boundary_rotation_divergence_deg
        << ";boundary_rotation_margin_deg=" << summary.boundary_rotation_margin_deg
        << ";boundary_rotation_pass="
        << static_cast<int>(summary.boundary_rotation_pass)
        << ";reentry_position_check_enabled="
        << static_cast<int>(config_.use_reentry_position_check)
        << ";reentry_rotation_check_enabled="
        << static_cast<int>(config_.use_reentry_rotation_check)
        << ";map_coverage_check_enabled="
        << static_cast<int>(config_.use_map_coverage_check)
        << ";map_support_check_enabled="
        << static_cast<int>(config_.use_map_support_check)
        << ";boundary_position_check_enabled="
        << static_cast<int>(config_.use_boundary_position_check)
        << ";boundary_rotation_check_enabled="
        << static_cast<int>(config_.use_boundary_rotation_check);
    return stream.str();
  }

  Config config_;
  bool armed_ = false;
  bool approved_ = false;
  double trigger_time_sec_ = -1.0;
  double residual_limit_ = 0.0;
  double match_ratio_limit_ = 0.0;
  double healthy_run_start_sec_ = -1.0;
  double observation_start_sec_ = -1.0;
  double last_observation_sec_ = -1.0;
  std::size_t consecutive_healthy_ = 0;
  bool candidate_summary_set_ = false;
  bool candidate_summary_healthy_ = false;
  double match_ratio_sum_ = 0.0;
  std::size_t observation_count_ = 0;
  std::size_t residual_pass_samples_ = 0;
  std::size_t match_ratio_pass_samples_ = 0;
  std::size_t jointly_healthy_samples_ = 0;
  std::size_t longest_consecutive_healthy_ = 0;
  double longest_healthy_duration_sec_ = 0.0;
  double minimum_residual_margin_ = std::numeric_limits<double>::quiet_NaN();
  double minimum_match_ratio_margin_ = std::numeric_limits<double>::quiet_NaN();
  CandidateSummaryDiagnostics candidate_summary_diagnostics_;
};

} // namespace recovery_validation
} // namespace fast_livo

#endif // HEALTHY_TAIL_VALIDATOR_H_
