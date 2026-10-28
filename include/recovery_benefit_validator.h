#ifndef RECOVERY_BENEFIT_VALIDATOR_H_
#define RECOVERY_BENEFIT_VALIDATOR_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fast_livo
{
namespace recovery_validation
{

struct BenefitConfig
{
  bool enabled = false;
  std::size_t minimum_observations = 20;
  double confidence_z = 1.645;
  double residual_relative_improvement = 0.05;
  double match_absolute_improvement = 0.02;
  double innovation_relative_improvement = 0.30;
  double minimum_improved_fraction_lower_bound = 0.40;

  void validate() const
  {
    if (!enabled) return;
    if (minimum_observations == 0 || confidence_z <= 0.0 ||
        residual_relative_improvement < 0.0 ||
        match_absolute_improvement < 0.0 ||
        innovation_relative_improvement < 0.0 ||
        minimum_improved_fraction_lower_bound < 0.0 ||
        minimum_improved_fraction_lower_bound > 1.0)
      throw std::invalid_argument("invalid paired recovery benefit config");
  }
};

class RecoveryBenefitValidator
{
public:
  explicit RecoveryBenefitValidator(const BenefitConfig &config = BenefitConfig())
      : config_(config)
  {
    config_.validate();
  }

  void observe(double incumbent_residual, double incumbent_match,
               double candidate_residual, double candidate_match,
               double incumbent_innovation =
                   std::numeric_limits<double>::quiet_NaN(),
               double candidate_innovation =
                   std::numeric_limits<double>::quiet_NaN())
  {
    if (!std::isfinite(incumbent_residual) || !std::isfinite(incumbent_match) ||
        !std::isfinite(candidate_residual) || !std::isfinite(candidate_match) ||
        incumbent_residual < 0.0 || candidate_residual < 0.0 ||
        incumbent_match < 0.0 || incumbent_match > 1.0 ||
        candidate_match < 0.0 || candidate_match > 1.0)
      return;
    const double denominator = std::max(incumbent_residual, 1e-12);
    const double residual_gain =
        (incumbent_residual - candidate_residual) / denominator;
    const double match_gain = candidate_match - incumbent_match;
    const bool innovation_valid = std::isfinite(incumbent_innovation) &&
        std::isfinite(candidate_innovation) && incumbent_innovation >= 0.0 &&
        candidate_innovation >= 0.0;
    double innovation_gain = -std::numeric_limits<double>::infinity();
    if (innovation_valid)
    {
      innovation_gain = (incumbent_innovation - candidate_innovation) /
          std::max(incumbent_innovation, 1e-12);
      innovation_gain_sum_ += innovation_gain;
      ++innovation_observation_count_;
    }
    residual_gain_sum_ += residual_gain;
    match_gain_sum_ += match_gain;
    ++observation_count_;
    if (residual_gain >= config_.residual_relative_improvement ||
        match_gain >= config_.match_absolute_improvement ||
        (innovation_valid && innovation_gain >=
            config_.innovation_relative_improvement))
      ++improved_count_;
  }

  std::size_t observationCount() const { return observation_count_; }
  std::size_t improvedCount() const { return improved_count_; }

  double improvedFraction() const
  {
    return observation_count_ == 0 ? 0.0 :
        static_cast<double>(improved_count_) /
            static_cast<double>(observation_count_);
  }

  double improvedFractionLowerBound() const
  {
    return wilsonLowerBound(improved_count_, observation_count_);
  }

  double meanResidualRelativeGain() const
  {
    return observation_count_ == 0 ?
        -std::numeric_limits<double>::infinity() :
        residual_gain_sum_ / static_cast<double>(observation_count_);
  }

  double meanMatchAbsoluteGain() const
  {
    return observation_count_ == 0 ?
        -std::numeric_limits<double>::infinity() :
        match_gain_sum_ / static_cast<double>(observation_count_);
  }

  double meanInnovationRelativeGain() const
  {
    return innovation_observation_count_ == 0 ?
        -std::numeric_limits<double>::infinity() :
        innovation_gain_sum_ /
            static_cast<double>(innovation_observation_count_);
  }

  bool approved() const
  {
    if (!config_.enabled) return true;
    if (observation_count_ < config_.minimum_observations) return false;
    const bool aggregate_gain =
        meanResidualRelativeGain() >= config_.residual_relative_improvement ||
        meanMatchAbsoluteGain() >= config_.match_absolute_improvement ||
        (innovation_observation_count_ >= config_.minimum_observations &&
         meanInnovationRelativeGain() >=
             config_.innovation_relative_improvement);
    return aggregate_gain && improvedFractionLowerBound() >=
        config_.minimum_improved_fraction_lower_bound;
  }

  // Return false only when approval is mathematically impossible even if
  // every remaining paired observation is maximally favorable. Residual
  // relative gain is upper-bounded by 1 because residuals are non-negative;
  // match gain is upper-bounded by 1 because match ratios are in [0, 1].
  bool approvalStillPossible(std::size_t remaining_observations) const
  {
    if (!config_.enabled) return true;
    const std::size_t final_count = observation_count_ + remaining_observations;
    if (final_count < config_.minimum_observations) return false;

    const std::size_t maximum_improved = improved_count_ + remaining_observations;
    const bool fraction_possible = wilsonLowerBound(
        maximum_improved, final_count) >=
        config_.minimum_improved_fraction_lower_bound;
    const double denominator = static_cast<double>(final_count);
    const bool aggregate_possible =
        (residual_gain_sum_ + static_cast<double>(remaining_observations)) /
                denominator >= config_.residual_relative_improvement ||
        (match_gain_sum_ + static_cast<double>(remaining_observations)) /
                denominator >= config_.match_absolute_improvement ||
        (innovation_observation_count_ + remaining_observations >=
             config_.minimum_observations &&
         (innovation_gain_sum_ + static_cast<double>(remaining_observations)) /
                 static_cast<double>(innovation_observation_count_ +
                                     remaining_observations) >=
             config_.innovation_relative_improvement);
    return fraction_possible && aggregate_possible;
  }

  std::string diagnosticFields() const
  {
    std::ostringstream out;
    out.precision(17);
    out << "paired_benefit_enabled=" << static_cast<int>(config_.enabled)
        << ";paired_benefit_approved=" << static_cast<int>(approved())
        << ";paired_benefit_observation_count=" << observation_count_
        << ";paired_benefit_minimum_observations="
        << config_.minimum_observations
        << ";paired_benefit_improved_count=" << improved_count_
        << ";paired_benefit_improved_fraction=" << improvedFraction()
        << ";paired_benefit_improved_fraction_lower_bound="
        << improvedFractionLowerBound()
        << ";paired_benefit_required_lower_bound="
        << config_.minimum_improved_fraction_lower_bound
        << ";paired_benefit_mean_residual_relative_gain="
        << meanResidualRelativeGain()
        << ";paired_benefit_required_residual_relative_gain="
        << config_.residual_relative_improvement
        << ";paired_benefit_mean_match_absolute_gain="
        << meanMatchAbsoluteGain()
        << ";paired_benefit_required_match_absolute_gain="
        << config_.match_absolute_improvement
        << ";paired_benefit_innovation_observation_count="
        << innovation_observation_count_
        << ";paired_benefit_mean_innovation_relative_gain="
        << meanInnovationRelativeGain()
        << ";paired_benefit_required_innovation_relative_gain="
        << config_.innovation_relative_improvement;
    return out.str();
  }

private:
  double wilsonLowerBound(std::size_t successes, std::size_t observations) const
  {
    if (observations == 0) return 0.0;
    const double n = static_cast<double>(observations);
    const double p = static_cast<double>(successes) / n;
    const double z2 = config_.confidence_z * config_.confidence_z;
    return std::max(0.0, (p + z2 / (2.0 * n) -
        config_.confidence_z * std::sqrt(
            (p * (1.0 - p) + z2 / (4.0 * n)) / n)) /
        (1.0 + z2 / n));
  }

  BenefitConfig config_;
  std::size_t observation_count_ = 0;
  std::size_t improved_count_ = 0;
  double residual_gain_sum_ = 0.0;
  double match_gain_sum_ = 0.0;
  std::size_t innovation_observation_count_ = 0;
  double innovation_gain_sum_ = 0.0;
};

} // namespace recovery_validation
} // namespace fast_livo

#endif // RECOVERY_BENEFIT_VALIDATOR_H_
