#include "recovery_benefit_validator.h"

#include <cassert>
#include <iostream>

int main()
{
  fast_livo::recovery_validation::BenefitConfig config;
  config.enabled = true;
  config.minimum_observations = 20;
  fast_livo::recovery_validation::RecoveryBenefitValidator improved(config);
  for (int i = 0; i < 30; ++i)
    improved.observe(0.10, 0.70, 0.08, 0.74);
  assert(improved.approved());
  assert(improved.improvedFractionLowerBound() > 0.40);
  assert(improved.approvalStillPossible(0));

  fast_livo::recovery_validation::RecoveryBenefitValidator equivalent(config);
  for (int i = 0; i < 30; ++i)
    equivalent.observe(0.10, 0.70, 0.099, 0.705);
  assert(!equivalent.approved());
  assert(!equivalent.approvalStillPossible(0));

  fast_livo::recovery_validation::RecoveryBenefitValidator recoverable(config);
  for (int i = 0; i < 10; ++i)
    recoverable.observe(0.10, 0.70, 0.099, 0.705);
  assert(recoverable.approvalStillPossible(20));

  fast_livo::recovery_validation::RecoveryBenefitValidator impossible(config);
  for (int i = 0; i < 30; ++i)
    impossible.observe(0.10, 0.70, 0.20, 0.10);
  assert(!impossible.approvalStillPossible(10));

  const std::size_t before_invalid = impossible.observationCount();
  impossible.observe(0.10, 1.2, 0.05, 0.90);
  assert(impossible.observationCount() == before_invalid);

  fast_livo::recovery_validation::RecoveryBenefitValidator innovation(config);
  for (int i = 0; i < 30; ++i)
    innovation.observe(0.10, 0.70, 0.11, 0.69, 2.0, 1.0);
  assert(innovation.approved());
  assert(innovation.meanInnovationRelativeGain() > 0.40);

  fast_livo::recovery_validation::RecoveryBenefitValidator sparse(config);
  for (int i = 0; i < 10; ++i)
    sparse.observe(0.10, 0.70, 0.05, 0.90);
  assert(!sparse.approved());

  std::cout << "RECOVERY_BENEFIT_VALIDATOR_TEST_OK" << std::endl;
  return 0;
}
