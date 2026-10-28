#ifndef SELECTIVE_RESIDUAL_GATE_H_
#define SELECTIVE_RESIDUAL_GATE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fast_livo
{
namespace selective_gating
{

struct Config
{
  double mad_multiplier = 2.5;
  double minimum_absolute_gate_m = 0.015;
  double maximum_absolute_gate_m = 0.15;
  double minimum_retained_fraction = 0.60;

  void validate() const
  {
    if (mad_multiplier < 0.0 || minimum_absolute_gate_m <= 0.0 ||
        maximum_absolute_gate_m < minimum_absolute_gate_m ||
        minimum_retained_fraction <= 0.0 || minimum_retained_fraction > 1.0)
      throw std::invalid_argument("invalid selective residual gate configuration");
  }
};

struct Decision
{
  std::vector<bool> keep;
  double median_m = 0.0;
  double sigma_m = 0.0;
  double threshold_m = 0.0;
  double retained_fraction = 1.0;
  std::size_t retained_count = 0;
};

inline double median(std::vector<double> values)
{
  if (values.empty()) return 0.0;
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (values.size() % 2 != 0) return upper;
  std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
  return 0.5 * (values[middle - 1] + upper);
}

inline Decision select(const std::vector<double> &absolute_residuals,
                       const Config &config = Config())
{
  config.validate();
  Decision decision;
  decision.keep.assign(absolute_residuals.size(), true);
  if (absolute_residuals.empty())
  {
    decision.retained_fraction = 0.0;
    return decision;
  }
  for (const double value : absolute_residuals)
  {
    if (!std::isfinite(value) || value < 0.0)
      throw std::invalid_argument("selective gate requires finite absolute residuals");
  }

  decision.median_m = median(absolute_residuals);
  std::vector<double> deviations;
  deviations.reserve(absolute_residuals.size());
  for (const double value : absolute_residuals)
    deviations.push_back(std::abs(value - decision.median_m));
  decision.sigma_m = 1.4826 * median(deviations);
  decision.threshold_m = std::min(
      config.maximum_absolute_gate_m,
      std::max(config.minimum_absolute_gate_m,
               decision.median_m + config.mad_multiplier * decision.sigma_m));

  std::vector<double> sorted = absolute_residuals;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t minimum_count = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(
             config.minimum_retained_fraction * static_cast<double>(sorted.size()))));
  decision.threshold_m = std::max(decision.threshold_m, sorted[minimum_count - 1]);

  decision.retained_count = 0;
  for (std::size_t index = 0; index < absolute_residuals.size(); ++index)
  {
    decision.keep[index] = absolute_residuals[index] <= decision.threshold_m;
    if (decision.keep[index]) ++decision.retained_count;
  }
  decision.retained_fraction = static_cast<double>(decision.retained_count) /
                               static_cast<double>(absolute_residuals.size());
  return decision;
}

} // namespace selective_gating
} // namespace fast_livo

#endif // SELECTIVE_RESIDUAL_GATE_H_
