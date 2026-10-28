#include "fault_injection_model.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
  using namespace fast_livo::fault_injection;
  Config config;
  config.start_sec = 10.0;
  config.end_sec = 20.0;
  config.range_scale_start = 1.0;
  config.range_scale_end = 1.1;
  config.ghost_fraction = 0.4;
  config.mode = LidarFaultMode::MIXED;
  config.seed = 42;
  Model model(config);

  assert(!model.active(9.999));
  assert(model.active(10.0));
  assert(model.active(20.0));
  assert(!model.active(20.001));
  assert(std::abs(model.rangeScale(15.0) - 1.05) < 1e-12);

  std::size_t selected = 0;
  for (std::size_t index = 0; index < 10000; ++index)
  {
    const bool first = model.ghostSelected(index, 7);
    const bool second = model.ghostSelected(index, 7);
    assert(first == second);
    if (first) ++selected;
  }
  assert(selected > 3700 && selected < 4300);

  double x = 1.0, y = 0.0, z = 0.0;
  model.corruptPoint(15.0, 0, 7, x, y, z);
  assert(std::isfinite(x) && std::isfinite(y) && std::isfinite(z));

  std::cout << "FAULT_INJECTION_MODEL_TEST_OK selected=" << selected << std::endl;
  return 0;
}
