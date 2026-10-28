#ifndef ROBUST_RESIDUAL_WEIGHT_H_
#define ROBUST_RESIDUAL_WEIGHT_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fast_livo
{
namespace robust_weighting
{

struct Config
{
  double mad_multiplier = 2.5;
  double minimum_transition_m = 0.015;
  double maximum_transition_m = 0.075;
  double minimum_weight = 0.10;

  void validate() const
  {
    if (mad_multiplier < 0.0 || minimum_transition_m <= 0.0 ||
        maximum_transition_m < minimum_transition_m ||
        minimum_weight <= 0.0 || minimum_weight > 1.0)
      throw std::invalid_argument("invalid robust residual weight configuration");
  }
};

struct Decision
{
  std::vector<double> weights;
  double median_m = 0.0;
  double sigma_m = 0.0;
  double transition_m = 0.0;
  double mean_weight = 1.0;
  double minimum_applied_weight = 1.0;
};

inline double median(std::vector<double> values)
{
  if (values.empty()) return 0.0;
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (values.size() % 2 != 0) return upper;
  std::nth_element(values.begin(), values.begin() + middle - 1,
                   values.begin() + middle);
  return 0.5 * (values[middle - 1] + upper);
}

// Adaptive Huber information weights. Residuals inside the robust transition
// retain their full information; only the tail is continuously attenuated.
inline Decision compute(const std::vector<double> &absolute_residuals,
                        const Config &config = Config())
{
  config.validate();
  Decision decision;
  decision.weights.assign(absolute_residuals.size(), 1.0);
  if (absolute_residuals.empty())
  {
    decision.mean_weight = 0.0;
    decision.minimum_applied_weight = 0.0;
    return decision;
  }
  for (const double value : absolute_residuals)
  {
    if (!std::isfinite(value) || value < 0.0)
      throw std::invalid_argument("robust weighting requires finite absolute residuals");
  }

  decision.median_m = median(absolute_residuals);
  std::vector<double> deviations;
  deviations.reserve(absolute_residuals.size());
  for (const double value : absolute_residuals)
    deviations.push_back(std::abs(value - decision.median_m));
  decision.sigma_m = 1.4826 * median(deviations);
  decision.transition_m = std::min(
      config.maximum_transition_m,
      std::max(config.minimum_transition_m,
               decision.median_m + config.mad_multiplier * decision.sigma_m));

  double weight_sum = 0.0;
  decision.minimum_applied_weight = 1.0;
  for (std::size_t index = 0; index < absolute_residuals.size(); ++index)
  {
    const double residual = absolute_residuals[index];
    const double huber_weight = residual <= decision.transition_m ?
        1.0 : decision.transition_m / residual;
    decision.weights[index] = std::max(config.minimum_weight, huber_weight);
    weight_sum += decision.weights[index];
    decision.minimum_applied_weight =
        std::min(decision.minimum_applied_weight, decision.weights[index]);
  }
  decision.mean_weight = weight_sum / static_cast<double>(decision.weights.size());
  return decision;
}

} // namespace robust_weighting
} // namespace fast_livo

#endif // ROBUST_RESIDUAL_WEIGHT_H_
