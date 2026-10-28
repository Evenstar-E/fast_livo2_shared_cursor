/*
 * Transaction state machine for bounded, replayable map updates.
 *
 * This component intentionally has no ROS, PCL, or Eigen dependencies so its
 * lifecycle semantics can be tested independently from the estimator.
 */

#ifndef MAP_TRANSACTION_STATE_H_
#define MAP_TRANSACTION_STATE_H_

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace fast_livo
{
namespace transactional_mapping
{

enum class Phase
{
  MONITORING,
  QUARANTINING,
  REPLAY_READY,
  REPLAYING,
  COMMITTED,
  ROLLED_BACK,
  OVERFLOWED
};

struct Config
{
  bool enabled = false;
  double checkpoint_start_sec = -1.0;
  double quarantine_start_sec = -1.0;
  double quarantine_end_sec = -1.0;
  double detection_delay_sec = 0.0;
  double recovery_tail_sec = 1.0;
  std::size_t max_buffered_events = 2000;

  void validate() const
  {
    if (!enabled) return;
    if (!std::isfinite(quarantine_start_sec) || !std::isfinite(quarantine_end_sec) ||
        !std::isfinite(detection_delay_sec) || !std::isfinite(recovery_tail_sec))
    {
      throw std::invalid_argument("transaction times must be finite");
    }
    if (quarantine_start_sec < 0.0 || quarantine_end_sec < quarantine_start_sec)
    {
      throw std::invalid_argument("invalid quarantine interval");
    }
    if (std::isfinite(checkpoint_start_sec) && checkpoint_start_sec >= 0.0 &&
        checkpoint_start_sec > quarantine_start_sec)
    {
      throw std::invalid_argument("checkpoint must not start after the quarantine interval");
    }
    if (detection_delay_sec < 0.0 || recovery_tail_sec < 0.0)
    {
      throw std::invalid_argument("detection delay and recovery tail must be non-negative");
    }
    if (max_buffered_events == 0)
    {
      throw std::invalid_argument("transaction buffer must hold at least one event");
    }
  }
};

struct Decision
{
  bool begin = false;
  bool buffer = false;
  bool detection_confirmed = false;
  bool trigger_replay = false;
};

class StateMachine
{
public:
  explicit StateMachine(const Config &config = Config()) : config_(config)
  {
    config_.validate();
  }

  const Config &config() const { return config_; }
  Phase phase() const { return phase_; }
  std::size_t bufferedEvents() const { return buffered_events_; }
  double checkpointStartTime() const
  {
    return config_.checkpoint_start_sec >= 0.0 ?
        config_.checkpoint_start_sec : quarantineStartTime();
  }
  double quarantineStartTime() const
  {
    return runtime_interval_ ? runtime_quarantine_start_sec_ : config_.quarantine_start_sec;
  }
  double quarantineEndTime() const
  {
    return runtime_interval_ ? runtime_quarantine_end_sec_ : config_.quarantine_end_sec;
  }
  double detectionTime() const
  {
    return runtime_interval_ ? runtime_detection_sec_ :
        config_.quarantine_start_sec + config_.detection_delay_sec;
  }
  double replayTime() const
  {
    return std::max(quarantineEndTime(), detectionTime()) + config_.recovery_tail_sec;
  }

  void setRuntimeInterval(double quarantine_start_sec, double quarantine_end_sec,
                          double detection_sec)
  {
    if (phase_ != Phase::MONITORING)
      throw std::logic_error("runtime interval must be set before recovery starts");
    if (!std::isfinite(quarantine_start_sec) || !std::isfinite(quarantine_end_sec) ||
        !std::isfinite(detection_sec) || quarantine_start_sec < 0.0 ||
        quarantine_end_sec < quarantine_start_sec || detection_sec < quarantine_start_sec)
      throw std::invalid_argument("invalid runtime quarantine interval");
    runtime_interval_ = true;
    runtime_quarantine_start_sec_ = quarantine_start_sec;
    runtime_quarantine_end_sec_ = quarantine_end_sec;
    runtime_detection_sec_ = detection_sec;
  }

  void updateRuntimeQuarantineEnd(double quarantine_end_sec)
  {
    if (!runtime_interval_ ||
        (phase_ != Phase::QUARANTINING && phase_ != Phase::REPLAY_READY))
      throw std::logic_error("runtime quarantine end can only update an active runtime interval");
    if (!std::isfinite(quarantine_end_sec) ||
        quarantine_end_sec < runtime_quarantine_start_sec_ ||
        quarantine_end_sec < runtime_detection_sec_)
      throw std::invalid_argument("invalid updated runtime quarantine end");
    runtime_quarantine_end_sec_ = quarantine_end_sec;
  }

  bool active() const
  {
    return phase_ == Phase::QUARANTINING || phase_ == Phase::REPLAY_READY || phase_ == Phase::REPLAYING;
  }

  bool isCorruptInterval(const double relative_time_sec) const
  {
    return config_.enabled && relative_time_sec >= quarantineStartTime() &&
           relative_time_sec <= quarantineEndTime();
  }

  Decision observe(const double relative_time_sec)
  {
    Decision decision;
    if (!config_.enabled || phase_ != Phase::MONITORING) return decision;
    if (!std::isfinite(relative_time_sec))
    {
      throw std::invalid_argument("observation time must be finite");
    }
    if (relative_time_sec < checkpointStartTime()) return decision;

    phase_ = Phase::QUARANTINING;
    decision.begin = true;
    decision.buffer = true;

    if (relative_time_sec >= detectionTime())
    {
      detection_reported_ = true;
      decision.detection_confirmed = true;
    }
    if (relative_time_sec > replayTime())
    {
      phase_ = Phase::REPLAY_READY;
      decision.trigger_replay = true;
    }
    return decision;
  }

  Decision advance(const double relative_time_sec)
  {
    if (phase_ == Phase::MONITORING) return observe(relative_time_sec);

    Decision decision;
    if (phase_ != Phase::QUARANTINING) return decision;
    if (!std::isfinite(relative_time_sec))
    {
      throw std::invalid_argument("observation time must be finite");
    }

    decision.buffer = true;
    if (!detection_reported_ && relative_time_sec >= detectionTime())
    {
      detection_reported_ = true;
      decision.detection_confirmed = true;
    }
    if (relative_time_sec > replayTime())
    {
      phase_ = Phase::REPLAY_READY;
      decision.trigger_replay = true;
    }
    return decision;
  }

  bool recordBufferedEvent()
  {
    if (phase_ != Phase::QUARANTINING && phase_ != Phase::REPLAY_READY) return false;
    ++buffered_events_;
    if (buffered_events_ > config_.max_buffered_events)
    {
      phase_ = Phase::OVERFLOWED;
      return false;
    }
    return true;
  }

  void markReplayStarted()
  {
    if (phase_ != Phase::REPLAY_READY)
    {
      throw std::logic_error("replay can only start when the transaction is replay-ready");
    }
    phase_ = Phase::REPLAYING;
  }

  void markCommitted()
  {
    if (phase_ != Phase::REPLAYING)
    {
      throw std::logic_error("only a replaying transaction can be committed");
    }
    phase_ = Phase::COMMITTED;
  }

  void markRolledBack()
  {
    if (!active() && phase_ != Phase::OVERFLOWED)
    {
      throw std::logic_error("no active transaction to roll back");
    }
    phase_ = Phase::ROLLED_BACK;
  }

  void resetForMonitoring()
  {
    if (phase_ != Phase::ROLLED_BACK)
      throw std::logic_error("only a rolled-back transaction can be re-armed");
    phase_ = Phase::MONITORING;
    buffered_events_ = 0;
    detection_reported_ = false;
    runtime_interval_ = false;
    runtime_quarantine_start_sec_ = -1.0;
    runtime_quarantine_end_sec_ = -1.0;
    runtime_detection_sec_ = -1.0;
  }

private:
  Config config_;
  Phase phase_ = Phase::MONITORING;
  std::size_t buffered_events_ = 0;
  bool detection_reported_ = false;
  bool runtime_interval_ = false;
  double runtime_quarantine_start_sec_ = -1.0;
  double runtime_quarantine_end_sec_ = -1.0;
  double runtime_detection_sec_ = -1.0;
};

} // namespace transactional_mapping
} // namespace fast_livo

#endif // MAP_TRANSACTION_STATE_H_
