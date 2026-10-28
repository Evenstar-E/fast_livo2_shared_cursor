#pragma once

#include <cmath>
#include <cstddef>
#include <string>

namespace fast_livo {
namespace replay {

// The incumbent snapshot is taken after the final buffered event. Comparing
// it with an earlier replay state silently understates or overstates the
// handover displacement when the platform is moving.
inline bool isCommitBoundaryEvent(std::size_t event_index,
                                  std::size_t event_count)
{
  return event_count != 0 && event_index == event_count - 1;
}

inline bool sameCommitBoundaryTime(double candidate_relative_time_sec,
                                   double incumbent_relative_time_sec)
{
  return std::isfinite(candidate_relative_time_sec) &&
         std::isfinite(incumbent_relative_time_sec) &&
         std::fabs(candidate_relative_time_sec -
                   incumbent_relative_time_sec) <= 1e-6;
}

// A handoff snapshot contains the state after its last completed fusion
// event. Its public boundary label may be one ULP earlier for ordering, but
// that label must never cause the completed event to be replayed twice.
inline bool isAfterCompletedCheckpointEvent(double event_relative_time_sec,
                                            double last_completed_time_sec)
{
  return std::isfinite(event_relative_time_sec) &&
         std::isfinite(last_completed_time_sec) &&
         event_relative_time_sec > last_completed_time_sec;
}

// One lidar period is far outside this. Do not widen it to hide a skipped frame.
constexpr double kSensorEpochToleranceSec = 2e-6;

struct SensorEventIdentity
{
  int type = 0;
  double absolute_time_sec = 0.0;
};

inline bool sensorEpochMatches(double observed_first_lidar_header_sec,
                               double sealed_epoch_sec)
{
  return std::isfinite(observed_first_lidar_header_sec) &&
         std::isfinite(sealed_epoch_sec) &&
         std::fabs(observed_first_lidar_header_sec - sealed_epoch_sec) <=
             kSensorEpochToleranceSec;
}

inline bool sameSensorEvent(const SensorEventIdentity &left,
                            const SensorEventIdentity &right)
{
  return left.type == right.type && left.type != 0 &&
         std::isfinite(left.absolute_time_sec) &&
         std::isfinite(right.absolute_time_sec) &&
         std::fabs(left.absolute_time_sec - right.absolute_time_sec) <=
             kSensorEpochToleranceSec;
}

// Relative time is absolute sensor time minus the sealed injector epoch.
// A later first lidar header is a different epoch and must not be used to
// slide an earlier event across the checkpoint boundary.
inline bool replayStartAccepted(double observed_first_lidar_header_sec,
                                double sealed_epoch_sec,
                                double absolute_sensor_time_sec,
                                double completed_event_time_sec,
                                const SensorEventIdentity &observed,
                                const SensorEventIdentity &expected_next)
{
  if (!sensorEpochMatches(observed_first_lidar_header_sec, sealed_epoch_sec) ||
      !sameSensorEvent(observed, expected_next))
    return false;
  const double relative_time = absolute_sensor_time_sec - sealed_epoch_sec;
  return isAfterCompletedCheckpointEvent(relative_time, completed_event_time_sec);
}

// Input identity is the synced cloud, image, and IMU. State hash is not part
// of it: the checkpoint state can match the correct event while the scan is
// already one frame later.
struct ReplayInputIdentity
{
  int type = 0;
  double absolute_time_sec = 0.0;
  std::string cur_hash;
  std::string image_hash;
  std::string imu_hash;
};

struct ReplayStartDecision
{
  bool accepted = false;
  const char *reason = "rejected";
};

inline bool replayInputDigestMatches(const ReplayInputIdentity &observed,
                                     const ReplayInputIdentity &expected)
{
  return !expected.cur_hash.empty() && !expected.image_hash.empty() &&
         !expected.imu_hash.empty() && observed.cur_hash == expected.cur_hash &&
         observed.image_hash == expected.image_hash &&
         observed.imu_hash == expected.imu_hash;
}

// Reject before any replay state or map update. A later first lidar header
// must not be subtracted so that a skipped frame's relative time lands on
// the sealed next event.
inline ReplayStartDecision acceptFirstReplayEvent(
    double observed_first_lidar_header_sec,
    double sealed_time_origin_sec,
    double completed_event_time_sec,
    const ReplayInputIdentity &observed,
    const ReplayInputIdentity &expected)
{
  ReplayStartDecision decision;
  if (!sensorEpochMatches(observed_first_lidar_header_sec,
                          sealed_time_origin_sec))
  {
    decision.reason = "epoch_mismatch";
    return decision;
  }
  if (observed.type == 0 || expected.type == 0 || observed.type != expected.type)
  {
    decision.reason = "event_type_mismatch";
    return decision;
  }
  if (!std::isfinite(observed.absolute_time_sec) ||
      !std::isfinite(expected.absolute_time_sec) ||
      std::fabs(observed.absolute_time_sec - expected.absolute_time_sec) >
          kSensorEpochToleranceSec)
  {
    decision.reason = "absolute_time_mismatch";
    return decision;
  }
  if (!replayInputDigestMatches(observed, expected))
  {
    decision.reason = "input_digest_mismatch";
    return decision;
  }
  const double relative_time =
      observed.absolute_time_sec - sealed_time_origin_sec;
  if (!isAfterCompletedCheckpointEvent(relative_time, completed_event_time_sec))
  {
    decision.reason = "not_after_completed_checkpoint";
    return decision;
  }
  decision.accepted = true;
  decision.reason = "accepted";
  return decision;
}

// The update runs only after the decision is accepted. A rejection leaves
// the incumbent object untouched.
template <class Update>
bool commitAcceptedReplayStart(const ReplayStartDecision &decision,
                               Update &&update)
{
  if (!decision.accepted) return false;
  update();
  return true;
}

}  // namespace replay
}  // namespace fast_livo
