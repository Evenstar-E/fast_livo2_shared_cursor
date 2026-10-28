#include "map_transaction_state.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using fast_livo::transactional_mapping::Config;
using fast_livo::transactional_mapping::Phase;
using fast_livo::transactional_mapping::StateMachine;

int main()
{
  {
    StateMachine disabled;
    const auto decision = disabled.advance(10.0);
    assert(!decision.begin && !decision.buffer && !decision.detection_confirmed && !decision.trigger_replay);
    assert(disabled.phase() == Phase::MONITORING);
  }

  Config config;
  config.enabled = true;
  config.quarantine_start_sec = 5.0;
  config.quarantine_end_sec = 8.0;
  config.recovery_tail_sec = 2.0;
  config.max_buffered_events = 3;

  {
    StateMachine transaction(config);
    assert(!transaction.advance(4.99).begin);

    auto decision = transaction.advance(5.0);
    assert(decision.begin && decision.buffer && decision.detection_confirmed && !decision.trigger_replay);
    assert(transaction.recordBufferedEvent());
    assert(transaction.isCorruptInterval(5.0));
    assert(transaction.isCorruptInterval(8.0));
    assert(!transaction.isCorruptInterval(8.01));

    decision = transaction.advance(10.0);
    assert(decision.buffer && !decision.trigger_replay);
    assert(transaction.recordBufferedEvent());

    decision = transaction.advance(10.01);
    assert(decision.buffer && decision.trigger_replay);
    assert(transaction.recordBufferedEvent());
    assert(transaction.phase() == Phase::REPLAY_READY);

    transaction.markReplayStarted();
    assert(transaction.phase() == Phase::REPLAYING);
    transaction.markCommitted();
    assert(transaction.phase() == Phase::COMMITTED);
  }

  {
    Config delayed = config;
    delayed.checkpoint_start_sec = 3.0;
    delayed.detection_delay_sec = 6.0;
    StateMachine transaction(delayed);
    assert(transaction.checkpointStartTime() == 3.0);
    assert(transaction.detectionTime() == 11.0);
    assert(transaction.replayTime() == 13.0);

    assert(!transaction.advance(2.99).begin);
    auto decision = transaction.advance(3.0);
    assert(decision.begin && decision.buffer && !decision.detection_confirmed);
    decision = transaction.advance(10.99);
    assert(decision.buffer && !decision.detection_confirmed && !decision.trigger_replay);
    decision = transaction.advance(11.0);
    assert(decision.buffer && decision.detection_confirmed && !decision.trigger_replay);
    decision = transaction.advance(11.01);
    assert(decision.buffer && !decision.detection_confirmed && !decision.trigger_replay);
    decision = transaction.advance(13.0);
    assert(decision.buffer && !decision.trigger_replay);
    decision = transaction.advance(13.01);
    assert(decision.buffer && decision.trigger_replay);
  }

  {
    bool rejected = false;
    try
    {
      Config invalid = config;
      invalid.checkpoint_start_sec = 5.1;
      StateMachine unused(invalid);
      (void)unused;
    }
    catch (const std::invalid_argument &)
    {
      rejected = true;
    }
    assert(rejected);
  }

  {
    StateMachine runtime(config);
    runtime.setRuntimeInterval(6.0, 14.0, 8.0);
    assert(runtime.checkpointStartTime() == 6.0);
    assert(runtime.quarantineStartTime() == 6.0);
    assert(runtime.quarantineEndTime() == 14.0);
    assert(runtime.detectionTime() == 8.0);
    assert(runtime.replayTime() == 16.0);
    assert(runtime.isCorruptInterval(7.0));
    assert(!runtime.isCorruptInterval(5.99));
    const auto decision = runtime.advance(8.0);
    assert(decision.begin && decision.detection_confirmed);
    runtime.updateRuntimeQuarantineEnd(12.0);
    assert(runtime.quarantineEndTime() == 12.0);
    assert(runtime.replayTime() == 14.0);
    runtime.markRolledBack();
    runtime.resetForMonitoring();
    assert(runtime.phase() == Phase::MONITORING);
    assert(runtime.quarantineStartTime() == config.quarantine_start_sec);
    assert(runtime.quarantineEndTime() == config.quarantine_end_sec);
  }

  {
    StateMachine overflow(config);
    overflow.advance(5.0);
    assert(overflow.recordBufferedEvent());
    assert(overflow.recordBufferedEvent());
    assert(overflow.recordBufferedEvent());
    assert(!overflow.recordBufferedEvent());
    assert(overflow.phase() == Phase::OVERFLOWED);
    overflow.markRolledBack();
    assert(overflow.phase() == Phase::ROLLED_BACK);
    overflow.resetForMonitoring();
    assert(overflow.phase() == Phase::MONITORING);
    assert(overflow.bufferedEvents() == 0);
    assert(overflow.quarantineStartTime() == config.quarantine_start_sec);
    assert(overflow.advance(5.0).begin);
  }

  {
    bool rejected = false;
    try
    {
      Config invalid = config;
      invalid.quarantine_end_sec = 4.0;
      StateMachine unused(invalid);
      (void)unused;
    }
    catch (const std::invalid_argument &)
    {
      rejected = true;
    }
    assert(rejected);
  }

  {
    bool rejected = false;
    try
    {
      Config invalid = config;
      invalid.detection_delay_sec = -0.1;
      StateMachine unused(invalid);
      (void)unused;
    }
    catch (const std::invalid_argument &)
    {
      rejected = true;
    }
    assert(rejected);
  }

  std::cout << "TRANSACTION_STATE_TEST_OK" << std::endl;
  return 0;
}
