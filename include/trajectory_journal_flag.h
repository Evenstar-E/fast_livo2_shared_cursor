#pragma once

namespace fast_livo::trajectory_journal
{
// 0 direct, 1 first production-replay post-update, 2 live transaction,
// 3 incumbent rebuild, 4 incumbent after-image, 5 a later replay sample.
// Flag 5 must not replace flag 1.
inline int selectFlag(bool incumbent_rebuild, bool production_replay,
                      bool transaction_active,
                      bool production_stamp_already_recorded)
{
  if (incumbent_rebuild) return 3;
  if (production_replay)
    return production_stamp_already_recorded ? 5 : 1;
  if (transaction_active) return 2;
  return 0;
}
}  // namespace fast_livo::trajectory_journal
