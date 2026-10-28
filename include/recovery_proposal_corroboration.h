#ifndef RECOVERY_PROPOSAL_CORROBORATION_H_
#define RECOVERY_PROPOSAL_CORROBORATION_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace fast_livo
{
namespace recovery_scheduling
{

struct ProposalCorroboration
{
  bool proposal_event = false;
  bool complete = false;
  double detection_sec = 0.0;
  double onset_lower_bound_sec = 0.0;
  double onset_upper_bound_sec = 0.0;
  double residual_score = 0.0;
  double match_score = 0.0;
  double innovation_score = 0.0;
  double raw_residual = 0.0;
};

inline bool parseStatusDouble(const std::string &status,
                              const std::string &key,
                              double *value)
{
  if (!value) return false;
  const std::string marker = key + "=";
  const std::size_t begin = status.find(marker);
  if (begin == std::string::npos) return false;
  const char *text = status.c_str() + begin + marker.size();
  char *end = nullptr;
  const double parsed = std::strtod(text, &end);
  if (end == text || !std::isfinite(parsed)) return false;
  *value = parsed;
  return true;
}

inline ProposalCorroboration parseProposalCorroboration(
    const std::string &status)
{
  ProposalCorroboration result;
  result.proposal_event =
      status.compare(0, std::string("recovery_proposed").size(),
                     "recovery_proposed") == 0 ||
      status.compare(0, std::string("recovery_proposal_observed").size(),
                     "recovery_proposal_observed") == 0;
  if (!result.proposal_event) return result;
  result.complete =
      parseStatusDouble(status, "detection_sec", &result.detection_sec) &&
      parseStatusDouble(status, "onset_lower_bound_sec",
                        &result.onset_lower_bound_sec) &&
      parseStatusDouble(status, "onset_upper_bound_sec",
                        &result.onset_upper_bound_sec) &&
      parseStatusDouble(status, "residual_score", &result.residual_score) &&
      parseStatusDouble(status, "match_score", &result.match_score) &&
      parseStatusDouble(status, "innovation_score", &result.innovation_score) &&
      parseStatusDouble(status, "raw_residual", &result.raw_residual);
  return result;
}

inline bool supportsMapRecovery(
    const ProposalCorroboration &evidence,
    const double minimum_raw_residual,
    const double maximum_innovation_to_lidar_ratio)
{
  if (!evidence.complete || minimum_raw_residual < 0.0 ||
      maximum_innovation_to_lidar_ratio <= 0.0)
    return false;
  const double lidar_score = std::max(evidence.residual_score,
                                      evidence.match_score);
  return evidence.raw_residual >= minimum_raw_residual &&
      lidar_score > 0.0 && evidence.innovation_score >= 0.0 &&
      evidence.innovation_score <=
          lidar_score * maximum_innovation_to_lidar_ratio;
}

inline bool corroboratesCandidateProposal(
    const ProposalCorroboration &evidence,
    const double candidate_detection_sec,
    const double candidate_onset_lower_bound_sec,
    const double candidate_onset_upper_bound_sec,
    const double maximum_detection_time_separation_sec)
{
  if (!evidence.complete ||
      !std::isfinite(candidate_detection_sec) ||
      !std::isfinite(candidate_onset_lower_bound_sec) ||
      !std::isfinite(candidate_onset_upper_bound_sec) ||
      !std::isfinite(maximum_detection_time_separation_sec) ||
      maximum_detection_time_separation_sec <= 0.0 ||
      evidence.onset_lower_bound_sec > evidence.onset_upper_bound_sec ||
      candidate_onset_lower_bound_sec > candidate_onset_upper_bound_sec)
    return false;
  const bool detection_window_matches =
      std::abs(evidence.detection_sec - candidate_detection_sec) <=
          maximum_detection_time_separation_sec;
  const bool onset_intervals_overlap =
      std::max(evidence.onset_lower_bound_sec,
               candidate_onset_lower_bound_sec) <=
      std::min(evidence.onset_upper_bound_sec,
               candidate_onset_upper_bound_sec);
  return detection_window_matches && onset_intervals_overlap;
}

} // namespace recovery_scheduling
} // namespace fast_livo

#endif // RECOVERY_PROPOSAL_CORROBORATION_H_
