#include "selective_residual_gate.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
  fast_livo::selective_gating::Config config;
  config.mad_multiplier = 2.5;
  config.minimum_absolute_gate_m = 0.01;
  config.maximum_absolute_gate_m = 0.15;
  config.minimum_retained_fraction = 0.60;

  std::vector<double> residuals = {
      0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.20, 0.30, 0.40, 0.50};
  const auto decision = fast_livo::selective_gating::select(residuals, config);
  assert(decision.retained_count == 6);
  assert(std::abs(decision.retained_fraction - 0.60) < 1e-12);
  assert(decision.keep[0] && decision.keep[5]);
  assert(!decision.keep[6] && !decision.keep[9]);

  std::vector<double> healthy = {0.010, 0.011, 0.012, 0.013, 0.014};
  const auto healthy_decision = fast_livo::selective_gating::select(healthy, config);
  assert(healthy_decision.retained_count == healthy.size());

  bool rejected_invalid = false;
  try
  {
    config.minimum_retained_fraction = 0.0;
    fast_livo::selective_gating::select(healthy, config);
  }
  catch (const std::invalid_argument &)
  {
    rejected_invalid = true;
  }
  assert(rejected_invalid);
  std::cout << "SELECTIVE_RESIDUAL_GATE_TEST_OK" << std::endl;
  return 0;
}
