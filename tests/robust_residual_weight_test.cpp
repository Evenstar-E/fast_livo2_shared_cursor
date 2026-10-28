#include "robust_residual_weight.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
  fast_livo::robust_weighting::Config config;
  config.mad_multiplier = 2.5;
  config.minimum_transition_m = 0.01;
  config.maximum_transition_m = 0.05;
  config.minimum_weight = 0.10;

  const std::vector<double> residuals = {
      0.008, 0.009, 0.010, 0.011, 0.012, 0.05, 0.10, 0.50};
  const auto decision = fast_livo::robust_weighting::compute(residuals, config);
  assert(decision.weights.size() == residuals.size());
  assert(decision.transition_m >= 0.01 && decision.transition_m <= 0.05);
  assert(decision.weights.front() == 1.0);
  assert(decision.weights.back() >= 0.10 && decision.weights.back() < 1.0);
  assert(decision.mean_weight > 0.0 && decision.mean_weight < 1.0);

  const std::vector<double> healthy = {0.010, 0.011, 0.012, 0.013, 0.014};
  const auto healthy_decision =
      fast_livo::robust_weighting::compute(healthy, config);
  for (const double weight : healthy_decision.weights) assert(weight == 1.0);

  bool rejected_invalid = false;
  try
  {
    config.minimum_weight = 0.0;
    fast_livo::robust_weighting::compute(healthy, config);
  }
  catch (const std::invalid_argument &)
  {
    rejected_invalid = true;
  }
  assert(rejected_invalid);
  std::cout << "ROBUST_RESIDUAL_WEIGHT_TEST_OK" << std::endl;
  return 0;
}
