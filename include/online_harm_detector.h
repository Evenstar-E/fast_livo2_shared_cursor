#ifndef ONLINE_HARM_DETECTOR_H_
#define ONLINE_HARM_DETECTOR_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace fast_livo
{
namespace harm_detection
{

struct Config
{
  bool enabled = false;
  bool two_timescale_enabled = false;
  double arming_delay_sec = 0.0;
  std::size_t baseline_window = 100;
  std::size_t trigger_consecutive = 3;
  double minimum_anomaly_duration_sec = 0.5;
  double residual_mad_multiplier = 6.0;
  double residual_increase_floor = 0.003;
  double match_mad_multiplier = 6.0;
  double match_drop_floor = 0.10;
  double innovation_mad_multiplier = 6.0;
  double innovation_increase_floor = 0.50;
  double residual_weight = 1.0;
  double match_weight = 1.0;
  double innovation_weight = 1.0;
  double strong_fused_score = 0.75;
  std::size_t minimum_strong_signals = 2;
  double dominant_signal_score = 2.5;
  std::size_t onset_guard_frames = 5;
  double onset_min_guard_sec = 0.5;
  double onset_max_guard_sec = 2.0;
  double weak_residual_fraction = 0.35;
  double weak_match_fraction = 0.35;
  double weak_innovation_fraction = 0.35;
  double weak_combined_score = 0.75;
  std::size_t weak_max_gap_frames = 1;
  double weak_lookback_sec = 2.5;
  double episode_merge_gap_sec = 12.0;
  double reference_window_sec = 8.0;
  double reference_gap_sec = 0.5;
  double current_window_sec = 2.0;
  bool two_timescale_innovation_two_sided = true;
  double two_timescale_minimum_residual_anchor_score = 0.35;
  double two_timescale_signal_support_score = 1.0;
  bool two_timescale_conservative_weak_onset_enabled = false;
  double two_timescale_conservative_weak_score_threshold = 1.0;

  void validate() const
  {
    if (!enabled) return;
    if (!std::isfinite(arming_delay_sec) || arming_delay_sec < 0.0)
      throw std::invalid_argument("harm detector arming delay must be finite and nonnegative");
    if (baseline_window < 10 || trigger_consecutive == 0 ||
        minimum_anomaly_duration_sec < 0.0)
      throw std::invalid_argument("harm detector needs a baseline and consecutive trigger");
    if (residual_mad_multiplier < 0.0 || residual_increase_floor <= 0.0 ||
        match_mad_multiplier < 0.0 || match_drop_floor <= 0.0 ||
        innovation_mad_multiplier < 0.0 || innovation_increase_floor <= 0.0)
      throw std::invalid_argument("harm detector thresholds must be positive");
    if (residual_weight < 0.0 || match_weight < 0.0 ||
        innovation_weight < 0.0 ||
        residual_weight + match_weight + innovation_weight <= 0.0 ||
        strong_fused_score <= 0.0 || minimum_strong_signals == 0 ||
        minimum_strong_signals > 3 || dominant_signal_score <= 1.0)
      throw std::invalid_argument("invalid multi-signal fusion configuration");
    if (onset_guard_frames == 0 || onset_min_guard_sec < 0.0 ||
        onset_max_guard_sec < onset_min_guard_sec)
      throw std::invalid_argument("invalid onset uncertainty guard");
    if (weak_residual_fraction <= 0.0 || weak_residual_fraction >= 1.0 ||
        weak_match_fraction <= 0.0 || weak_match_fraction >= 1.0 ||
        weak_innovation_fraction <= 0.0 || weak_innovation_fraction >= 1.0 ||
        weak_combined_score <= 0.0 || weak_lookback_sec <= 0.0)
      throw std::invalid_argument("invalid weak-evidence change-point configuration");
    if (episode_merge_gap_sec < 0.0)
      throw std::invalid_argument("invalid degradation episode merge gap");
    if (two_timescale_enabled &&
        (reference_window_sec <= 0.0 || reference_gap_sec < 0.0 ||
         current_window_sec <= 0.0 ||
         two_timescale_minimum_residual_anchor_score < 0.0 ||
         two_timescale_signal_support_score <= 0.0 ||
         two_timescale_conservative_weak_score_threshold <= 0.0))
      throw std::invalid_argument("invalid two-timescale detector windows");
  }
};

struct Decision
{
  bool valid_observation = true;
  bool baseline_ready = false;
  bool anomalous = false;
  bool weak_evidence = false;
  bool triggered = false;
  double estimated_onset_sec = -1.0;
  double onset_lower_bound_sec = -1.0;
  double onset_upper_bound_sec = -1.0;
  double onset_uncertainty_sec = 0.0;
  double onset_confidence = 0.0;
  std::string onset_interval_semantics = "causal_evidence_envelope";
  double anomaly_duration_sec = 0.0;
  bool change_point_found = false;
  bool precursor_supported = false;
  double change_point_lower_sec = -1.0;
  double change_point_upper_sec = -1.0;
  std::size_t weak_evidence_samples = 0;
  double residual_threshold = 0.0;
  double match_ratio_threshold = 0.0;
  double innovation_threshold = 0.0;
  double residual_median = 0.0;
  double residual_sigma = 0.0;
  double match_ratio_median = 0.0;
  double match_ratio_sigma = 0.0;
  double innovation_median = 0.0;
  double innovation_sigma = 0.0;
  double residual_score = 0.0;
  double match_score = 0.0;
  double innovation_score = 0.0;
  double fused_score = 0.0;
  std::size_t strong_signal_count = 0;
  std::size_t episode_id = 0;
  std::size_t proposal_index_in_episode = 0;
  bool merged_with_previous_proposal = false;
  double episode_lower_bound_sec = -1.0;
  double episode_upper_bound_sec = -1.0;
};

class OnlineHarmDetector
{
public:
  explicit OnlineHarmDetector(const Config &config = Config()) : config_(config)
  {
    config_.validate();
  }

  Decision observe(double time_sec, double mean_abs_residual, double match_ratio,
                   double normalized_innovation = 0.0,
                   bool allow_trigger = true)
  {
    Decision decision;
    if (!std::isfinite(time_sec) || !std::isfinite(mean_abs_residual) ||
        !std::isfinite(match_ratio) || !std::isfinite(normalized_innovation) ||
        mean_abs_residual < 0.0 || normalized_innovation < 0.0 ||
        match_ratio < 0.0 || match_ratio > 1.0)
    {
      // A temporarily empty scan has no valid residual/match statistic.  It
      // is neither healthy baseline evidence nor a detector exception: keep
      // the previous detector state and let the caller freeze anchor rotation
      // until a valid observation arrives.
      decision.valid_observation = false;
      return decision;
    }

    if (last_sample_time_sec_ >= 0.0 && time_sec > last_sample_time_sec_)
    {
      sample_intervals_sec_.push_back(time_sec - last_sample_time_sec_);
      while (sample_intervals_sec_.size() > config_.baseline_window)
        sample_intervals_sec_.pop_front();
    }
    last_sample_time_sec_ = time_sec;

    if (!config_.enabled || triggered_) return decision;
    if (time_sec < config_.arming_delay_sec)
    {
      // Startup motion and map bootstrap can look like a persistent sensor
      // fault to a two-window contrast.  Do not let those samples seed either
      // reference window; monitoring starts from a fresh causal history after
      // the configured deployment warmup and consumes no fault label.
      baseline_.clear();
      long_history_.clear();
      evidence_history_.clear();
      sample_intervals_sec_.clear();
      consecutive_anomalies_ = 0;
      candidate_onset_sec_ = -1.0;
      last_sample_time_sec_ = -1.0;
      return decision;
    }
    if (config_.two_timescale_enabled)
      return observeTwoTimescale(
          time_sec, mean_abs_residual, match_ratio, normalized_innovation,
          allow_trigger);
    if (baseline_.size() < config_.baseline_window)
    {
      pushBaseline(mean_abs_residual, match_ratio, normalized_innovation);
      return decision;
    }

    decision.baseline_ready = true;
    decision.residual_median = median(Signal::RESIDUAL);
    decision.match_ratio_median = median(Signal::MATCH_RATIO);
    decision.innovation_median = median(Signal::INNOVATION);
    decision.residual_sigma = 1.4826 * mad(
        decision.residual_median, Signal::RESIDUAL);
    decision.match_ratio_sigma = 1.4826 * mad(
        decision.match_ratio_median, Signal::MATCH_RATIO);
    decision.innovation_sigma = 1.4826 * mad(
        decision.innovation_median, Signal::INNOVATION);
    decision.residual_threshold = decision.residual_median +
        std::max(config_.residual_mad_multiplier * decision.residual_sigma,
                 config_.residual_increase_floor);
    decision.match_ratio_threshold = decision.match_ratio_median -
        std::max(config_.match_mad_multiplier * decision.match_ratio_sigma,
                 config_.match_drop_floor);
    decision.innovation_threshold = decision.innovation_median +
        std::max(config_.innovation_mad_multiplier * decision.innovation_sigma,
                 config_.innovation_increase_floor);
    const double residual_scale = decision.residual_threshold - decision.residual_median;
    const double match_scale = decision.match_ratio_median - decision.match_ratio_threshold;
    decision.residual_score = std::max(
        0.0, (mean_abs_residual - decision.residual_median) / residual_scale);
    decision.match_score = std::max(
        0.0, (decision.match_ratio_median - match_ratio) / match_scale);
    const double innovation_scale =
        decision.innovation_threshold - decision.innovation_median;
    decision.innovation_score = std::max(
        0.0, (normalized_innovation - decision.innovation_median) /
        innovation_scale);
    decision.strong_signal_count =
        static_cast<std::size_t>(decision.residual_score > 1.0) +
        static_cast<std::size_t>(decision.match_score > 1.0) +
        static_cast<std::size_t>(decision.innovation_score > 1.0);
    const double weight_sum = config_.residual_weight + config_.match_weight +
                              config_.innovation_weight;
    decision.fused_score =
        (config_.residual_weight * decision.residual_score +
         config_.match_weight * decision.match_score +
         config_.innovation_weight * decision.innovation_score) / weight_sum;
    const double maximum_score = std::max(
        decision.residual_score,
        std::max(decision.match_score, decision.innovation_score));
    decision.anomalous =
        (decision.strong_signal_count >= config_.minimum_strong_signals &&
         decision.fused_score >= config_.strong_fused_score) ||
        (maximum_score >= config_.dominant_signal_score &&
         decision.strong_signal_count >= 1 &&
         ((decision.residual_score >= config_.weak_residual_fraction) +
          (decision.match_score >= config_.weak_match_fraction) +
          (decision.innovation_score >= config_.weak_innovation_fraction) >= 2));
    const bool weak_evidence =
        (decision.residual_score >= config_.weak_residual_fraction ||
         decision.match_score >= config_.weak_match_fraction ||
         decision.innovation_score >= config_.weak_innovation_fraction) &&
        decision.fused_score >= config_.weak_combined_score / 3.0;
    decision.weak_evidence = weak_evidence;
    evidence_history_.push_back(
        {time_sec, mean_abs_residual, match_ratio, normalized_innovation,
         weak_evidence});
    while (!evidence_history_.empty() &&
           time_sec - evidence_history_.front().time_sec > config_.weak_lookback_sec)
      evidence_history_.pop_front();
    // Publish the active weak-evidence suffix before a strong trigger occurs.
    // Rolling checkpoint retention needs this signal to protect the last
    // healthy anchor across the same short gaps accepted by change-point
    // estimation. Waiting until estimateChangePoint() runs at trigger time is
    // too late: that anchor may already have been rotated out.
    decision.weak_evidence_samples = weakEvidenceSuffixCount();

    // A scheduler-suppressed proposal still belongs to the anomaly episode
    // already in progress. Keep observing, but require a causal healthy
    // transition before the detector may emit another proposal. This changes
    // latch semantics only; all frozen score and timing thresholds are intact.
    if (suppressed_rearm_pending_)
    {
      consecutive_anomalies_ = 0;
      candidate_onset_sec_ = -1.0;
      if (!decision.anomalous) suppressed_rearm_pending_ = false;
      return decision;
    }

    if (decision.anomalous)
    {
      if (consecutive_anomalies_ == 0) candidate_onset_sec_ = time_sec;
      ++consecutive_anomalies_;
      decision.anomaly_duration_sec =
          std::max(0.0, time_sec - candidate_onset_sec_);
      if (consecutive_anomalies_ >= config_.trigger_consecutive &&
          decision.anomaly_duration_sec >=
              config_.minimum_anomaly_duration_sec)
      {
        if (!allow_trigger) return decision;
        triggered_ = true;
        decision.triggered = true;
        decision.estimated_onset_sec = candidate_onset_sec_;
        const double timing_guard = static_cast<double>(config_.onset_guard_frames) *
                                    medianSampleInterval();
        const double fallback_guard = std::min(
            config_.onset_max_guard_sec,
            std::max(config_.onset_min_guard_sec, timing_guard));
        const double fallback_lower = std::max(
            0.0, decision.estimated_onset_sec - fallback_guard);
        estimateChangePoint(decision);
        decision.onset_lower_bound_sec = fallback_lower;
        decision.precursor_supported = decision.change_point_found &&
            decision.weak_evidence_samples > config_.trigger_consecutive &&
            decision.change_point_upper_sec < decision.estimated_onset_sec;
        if (decision.precursor_supported)
        {
          decision.estimated_onset_sec = decision.change_point_upper_sec;
          // The same timing uncertainty used by the abrupt-fault fallback is
          // anchored at the statistically observed change-point bracket.  This
          // creates a lower confidence boundary that expands with precursor
          // duration, while an abrupt trigger retains the original guard.
          const double adaptive_lower = std::max(
              0.0, decision.change_point_lower_sec - fallback_guard);
          decision.onset_lower_bound_sec = std::min(
              fallback_lower, adaptive_lower);
        }
        decision.onset_uncertainty_sec =
            decision.estimated_onset_sec - decision.onset_lower_bound_sec;
        decision.onset_upper_bound_sec = decision.estimated_onset_sec;
        assignEpisode(time_sec, decision);
        decision.onset_confidence = onsetConfidence(decision);
      }
    }
    else
    {
      consecutive_anomalies_ = 0;
      candidate_onset_sec_ = -1.0;
      // Preserve the adaptive baseline for sub-trigger scene changes. The P1
      // long-horizon branch is developed independently from this fast branch.
      pushBaseline(mean_abs_residual, match_ratio, normalized_innovation);
    }
    return decision;
  }

  bool triggered() const { return triggered_; }
  bool awaitingHealthyRearm() const { return suppressed_rearm_pending_; }
  std::size_t baselineSize() const
  {
    return config_.two_timescale_enabled ? long_history_.size() : baseline_.size();
  }

  // A rejected recovery has been reconstructed with the unmodified estimator.
  // Relearn the local healthy regime before permitting another trigger; using
  // the pre-transition baseline would immediately reproduce the same proposal.
  void resetForMonitoring()
  {
    baseline_.clear();
    evidence_history_.clear();
    if (!config_.two_timescale_enabled)
    {
      sample_intervals_sec_.clear();
      long_history_.clear();
      last_sample_time_sec_ = -1.0;
    }
    consecutive_anomalies_ = 0;
    candidate_onset_sec_ = -1.0;
    triggered_ = false;
    suppressed_rearm_pending_ = false;
  }

  // Release a proposal latch rejected by the recovery scheduler without
  // erasing the two-timescale history. Persistent samples from the same
  // degradation interval cannot create duplicate proposals; the first healthy
  // observation re-arms the detector for a later, distinct interval.
  void suppressUntilHealthyTransition()
  {
    consecutive_anomalies_ = 0;
    candidate_onset_sec_ = -1.0;
    triggered_ = false;
    suppressed_rearm_pending_ = true;
  }

  struct Sample
  {
    double time_sec = -1.0;
    double residual = 0.0;
    double match_ratio = 0.0;
    double innovation = 0.0;
    bool weak_evidence = false;
  };

  struct Snapshot
  {
    std::deque<Sample> baseline;
    std::deque<Sample> long_history;
    std::deque<Sample> evidence_history;
    std::deque<double> sample_intervals_sec;
    std::size_t consecutive_anomalies = 0;
    double candidate_onset_sec = -1.0;
    double last_sample_time_sec = -1.0;
    bool triggered = false;
    bool suppressed_rearm_pending = false;
    std::size_t episode_id = 0;
    std::size_t episode_proposal_count = 0;
    double last_episode_detection_sec = -1.0;
    double last_episode_lower_bound_sec = -1.0;
    double last_episode_upper_bound_sec = -1.0;
  };

  Snapshot createSnapshot() const
  {
    Snapshot snapshot;
    snapshot.baseline = baseline_;
    snapshot.long_history = long_history_;
    snapshot.evidence_history = evidence_history_;
    snapshot.sample_intervals_sec = sample_intervals_sec_;
    snapshot.consecutive_anomalies = consecutive_anomalies_;
    snapshot.candidate_onset_sec = candidate_onset_sec_;
    snapshot.last_sample_time_sec = last_sample_time_sec_;
    snapshot.triggered = triggered_;
    snapshot.suppressed_rearm_pending = suppressed_rearm_pending_;
    snapshot.episode_id = episode_id_;
    snapshot.episode_proposal_count = episode_proposal_count_;
    snapshot.last_episode_detection_sec = last_episode_detection_sec_;
    snapshot.last_episode_lower_bound_sec = last_episode_lower_bound_sec_;
    snapshot.last_episode_upper_bound_sec = last_episode_upper_bound_sec_;
    return snapshot;
  }

  void restoreSnapshot(const Snapshot &snapshot)
  {
    baseline_ = snapshot.baseline;
    long_history_ = snapshot.long_history;
    evidence_history_ = snapshot.evidence_history;
    sample_intervals_sec_ = snapshot.sample_intervals_sec;
    consecutive_anomalies_ = snapshot.consecutive_anomalies;
    candidate_onset_sec_ = snapshot.candidate_onset_sec;
    last_sample_time_sec_ = snapshot.last_sample_time_sec;
    triggered_ = snapshot.triggered;
    suppressed_rearm_pending_ = snapshot.suppressed_rearm_pending;
    episode_id_ = snapshot.episode_id;
    episode_proposal_count_ = snapshot.episode_proposal_count;
    last_episode_detection_sec_ = snapshot.last_episode_detection_sec;
    last_episode_lower_bound_sec_ = snapshot.last_episode_lower_bound_sec;
    last_episode_upper_bound_sec_ = snapshot.last_episode_upper_bound_sec;
  }

private:

  enum class Signal { RESIDUAL, MATCH_RATIO, INNOVATION };

  static double vectorMedian(std::vector<double> values)
  {
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2) return values[middle];
    const double upper = values[middle];
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    return 0.5 * (upper + values[middle - 1]);
  }

  static double vectorMad(const std::vector<double> &values, double center)
  {
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values) deviations.push_back(std::abs(value - center));
    return vectorMedian(deviations);
  }

  static std::vector<double> valuesFor(
      const std::vector<Sample> &samples, Signal signal)
  {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const Sample &sample : samples)
      values.push_back(signalValue(sample, signal));
    return values;
  }

  Decision observeTwoTimescale(double time_sec, double residual,
                               double match_ratio, double innovation,
                               bool allow_trigger)
  {
    Decision decision;
    long_history_.push_back(
        {time_sec, residual, match_ratio, innovation, false});
    const double history_sec = config_.reference_window_sec +
        config_.reference_gap_sec + config_.current_window_sec;
    while (!long_history_.empty() &&
           time_sec - long_history_.front().time_sec > history_sec)
      long_history_.pop_front();

    const double reference_end = time_sec - config_.current_window_sec -
                                 config_.reference_gap_sec;
    const double reference_start = reference_end - config_.reference_window_sec;
    const double current_start = time_sec - config_.current_window_sec;
    std::vector<Sample> reference;
    std::vector<Sample> current;
    for (const Sample &sample : long_history_)
    {
      if (sample.time_sec >= reference_start && sample.time_sec <= reference_end)
        reference.push_back(sample);
      if (sample.time_sec >= current_start) current.push_back(sample);
    }
    if (reference.size() < 2 || current.size() < 2 ||
        reference.back().time_sec - reference.front().time_sec <
            0.75 * config_.reference_window_sec ||
        current.back().time_sec - current.front().time_sec <
            0.75 * config_.current_window_sec)
      return decision;

    decision.baseline_ready = true;
    const std::vector<double> reference_residual =
        valuesFor(reference, Signal::RESIDUAL);
    const std::vector<double> reference_match =
        valuesFor(reference, Signal::MATCH_RATIO);
    const std::vector<double> reference_innovation =
        valuesFor(reference, Signal::INNOVATION);
    decision.residual_median = vectorMedian(reference_residual);
    decision.match_ratio_median = vectorMedian(reference_match);
    decision.innovation_median = vectorMedian(reference_innovation);
    decision.residual_sigma = 1.4826 * vectorMad(
        reference_residual, decision.residual_median);
    decision.match_ratio_sigma = 1.4826 * vectorMad(
        reference_match, decision.match_ratio_median);
    decision.innovation_sigma = 1.4826 * vectorMad(
        reference_innovation, decision.innovation_median);
    const double residual_scale = std::max(
        config_.residual_mad_multiplier * decision.residual_sigma,
        config_.residual_increase_floor);
    const double match_scale = std::max(
        config_.match_mad_multiplier * decision.match_ratio_sigma,
        config_.match_drop_floor);
    const double innovation_scale = std::max(
        config_.innovation_mad_multiplier * decision.innovation_sigma,
        config_.innovation_increase_floor);
    decision.residual_threshold = decision.residual_median + residual_scale;
    decision.match_ratio_threshold = decision.match_ratio_median - match_scale;
    decision.innovation_threshold = decision.innovation_median + innovation_scale;
    decision.residual_score = std::max(
        0.0, (vectorMedian(valuesFor(current, Signal::RESIDUAL)) -
              decision.residual_median) / residual_scale);
    decision.match_score = std::max(
        0.0, (decision.match_ratio_median -
              vectorMedian(valuesFor(current, Signal::MATCH_RATIO))) / match_scale);
    const double innovation_shift =
        vectorMedian(valuesFor(current, Signal::INNOVATION)) -
        decision.innovation_median;
    decision.innovation_score = config_.two_timescale_innovation_two_sided ?
        std::abs(innovation_shift) / innovation_scale :
        std::max(0.0, innovation_shift / innovation_scale);
    decision.strong_signal_count =
        static_cast<std::size_t>(decision.residual_score >=
                                 config_.two_timescale_signal_support_score) +
        static_cast<std::size_t>(decision.match_score >=
                                 config_.two_timescale_signal_support_score) +
        static_cast<std::size_t>(decision.innovation_score >=
                                 config_.two_timescale_signal_support_score);
    const double weight_sum = config_.residual_weight + config_.match_weight +
                              config_.innovation_weight;
    decision.fused_score =
        (config_.residual_weight * decision.residual_score +
         config_.match_weight * decision.match_score +
         config_.innovation_weight * decision.innovation_score) / weight_sum;
    const double maximum_score = std::max(
        decision.residual_score,
        std::max(decision.match_score, decision.innovation_score));
    decision.anomalous = decision.residual_score >=
        config_.two_timescale_minimum_residual_anchor_score && (
        (decision.strong_signal_count >= config_.minimum_strong_signals &&
         decision.fused_score >= config_.strong_fused_score) ||
        (maximum_score >= config_.dominant_signal_score &&
         ((decision.residual_score >= config_.weak_residual_fraction) +
          (decision.match_score >= config_.weak_match_fraction) +
          (decision.innovation_score >= config_.weak_innovation_fraction) >= 2)));
    decision.weak_evidence =
        (decision.residual_score >= config_.weak_residual_fraction ||
         decision.match_score >= config_.weak_match_fraction ||
         decision.innovation_score >= config_.weak_innovation_fraction);
    if (suppressed_rearm_pending_)
    {
      consecutive_anomalies_ = 0;
      candidate_onset_sec_ = -1.0;
      if (!decision.anomalous) suppressed_rearm_pending_ = false;
      return decision;
    }
    if (!decision.anomalous)
    {
      consecutive_anomalies_ = 0;
      candidate_onset_sec_ = -1.0;
      return decision;
    }
    if (consecutive_anomalies_ == 0) candidate_onset_sec_ = time_sec;
    ++consecutive_anomalies_;
    decision.anomaly_duration_sec = time_sec - candidate_onset_sec_;
    if (consecutive_anomalies_ < config_.trigger_consecutive ||
        decision.anomaly_duration_sec < config_.minimum_anomaly_duration_sec)
      return decision;

    if (!allow_trigger) return decision;
    triggered_ = true;
    decision.triggered = true;
    const double nominal_lower_bound = std::max(
        long_history_.front().time_sec,
        candidate_onset_sec_ - config_.current_window_sec);
    decision.onset_upper_bound_sec = candidate_onset_sec_;
    decision.estimated_onset_sec = 0.5 *
        (nominal_lower_bound + decision.onset_upper_bound_sec);
    decision.onset_lower_bound_sec = nominal_lower_bound;
    if (config_.two_timescale_conservative_weak_onset_enabled &&
        maximum_score <
            config_.two_timescale_conservative_weak_score_threshold)
    {
      // A weak two-window contrast can be the delayed footprint of a gradual
      // fault.  Keep the point estimate unchanged, but widen only the recovery
      // side of the interval across the complete causal support horizon.  The
      // feature is opt-in so the frozen P1 point/interval protocol is unchanged.
      const double causal_support_horizon = config_.reference_window_sec +
          config_.reference_gap_sec + config_.current_window_sec +
          config_.minimum_anomaly_duration_sec;
      // This is a temporal uncertainty bound, not an observed-sample bound:
      // a slowly accumulating fault may begin before the oldest sample still
      // needed for the active reference/current comparison.
      decision.onset_lower_bound_sec = std::max(
          0.0, candidate_onset_sec_ - causal_support_horizon);
      decision.onset_interval_semantics =
          "weak_evidence_causal_support_envelope";
    }
    decision.onset_uncertainty_sec = std::max(
        decision.estimated_onset_sec - decision.onset_lower_bound_sec,
        decision.onset_upper_bound_sec - decision.estimated_onset_sec);
    decision.change_point_found = true;
    decision.change_point_lower_sec = decision.onset_lower_bound_sec;
    decision.change_point_upper_sec = decision.onset_upper_bound_sec;
    assignEpisode(time_sec, decision);
    decision.onset_confidence = onsetConfidence(decision);
    return decision;
  }

  void pushBaseline(double residual, double match_ratio, double innovation)
  {
    baseline_.push_back({-1.0, residual, match_ratio, innovation, false});
    while (baseline_.size() > config_.baseline_window) baseline_.pop_front();
  }

  std::size_t weakEvidenceSuffixCount() const
  {
    std::size_t evidence_count = 0;
    std::size_t gap_frames = 0;
    for (std::size_t reverse = evidence_history_.size(); reverse > 0; --reverse)
    {
      const Sample &sample = evidence_history_[reverse - 1];
      if (sample.weak_evidence)
      {
        ++evidence_count;
        gap_frames = 0;
      }
      else
      {
        ++gap_frames;
        if (gap_frames > config_.weak_max_gap_frames) break;
      }
    }
    return evidence_count;
  }

  static double signalValue(const Sample &sample, Signal signal)
  {
    if (signal == Signal::RESIDUAL) return sample.residual;
    if (signal == Signal::MATCH_RATIO) return sample.match_ratio;
    return sample.innovation;
  }

  double median(Signal signal) const
  {
    std::vector<double> values;
    values.reserve(baseline_.size());
    for (const Sample &sample : baseline_)
      values.push_back(signalValue(sample, signal));
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2) return values[middle];
    const double upper = values[middle];
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    return 0.5 * (upper + values[middle - 1]);
  }

  double mad(double center, Signal signal) const
  {
    std::vector<double> deviations;
    deviations.reserve(baseline_.size());
    for (const Sample &sample : baseline_)
      deviations.push_back(std::abs(signalValue(sample, signal) - center));
    const std::size_t middle = deviations.size() / 2;
    std::nth_element(deviations.begin(), deviations.begin() + middle, deviations.end());
    if (deviations.size() % 2) return deviations[middle];
    const double upper = deviations[middle];
    std::nth_element(deviations.begin(), deviations.begin() + middle - 1, deviations.end());
    return 0.5 * (upper + deviations[middle - 1]);
  }

  double medianSampleInterval() const
  {
    if (sample_intervals_sec_.empty()) return 0.0;
    std::vector<double> values(sample_intervals_sec_.begin(), sample_intervals_sec_.end());
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2) return values[middle];
    const double upper = values[middle];
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    return 0.5 * (upper + values[middle - 1]);
  }

  void estimateChangePoint(Decision &decision) const
  {
    if (evidence_history_.empty()) return;

    std::size_t earliest_weak = evidence_history_.size();
    std::size_t evidence_count = 0;
    std::size_t gap_frames = 0;
    for (std::size_t reverse = evidence_history_.size(); reverse > 0; --reverse)
    {
      const std::size_t index = reverse - 1;
      const Sample &sample = evidence_history_[index];
      if (decision.estimated_onset_sec - sample.time_sec > config_.weak_lookback_sec)
        break;
      if (sample.weak_evidence)
      {
        earliest_weak = index;
        ++evidence_count;
        gap_frames = 0;
      }
      else if (earliest_weak != evidence_history_.size())
      {
        ++gap_frames;
        if (gap_frames > config_.weak_max_gap_frames) break;
      }
    }

    if (earliest_weak == evidence_history_.size()) return;
    decision.change_point_found = true;
    decision.change_point_upper_sec = evidence_history_[earliest_weak].time_sec;
    decision.change_point_lower_sec = decision.change_point_upper_sec;
    if (earliest_weak > 0)
      decision.change_point_lower_sec = evidence_history_[earliest_weak - 1].time_sec;
    else
      decision.change_point_lower_sec = std::max(
          0.0, decision.change_point_upper_sec - medianSampleInterval());
    decision.weak_evidence_samples = evidence_count;
  }

  static double clamp01(double value)
  {
    return std::max(0.0, std::min(1.0, value));
  }

  double onsetConfidence(const Decision &decision) const
  {
    const double evidence = clamp01(
        decision.fused_score / std::max(config_.strong_fused_score, 1e-12));
    const double agreement = static_cast<double>(decision.strong_signal_count) / 3.0;
    const double persistence = clamp01(
        static_cast<double>(consecutive_anomalies_) /
        static_cast<double>(config_.trigger_consecutive));
    const double interval_width = std::max(
        0.0, decision.onset_upper_bound_sec - decision.onset_lower_bound_sec);
    const double compactness = std::exp(
        -interval_width / std::max(config_.onset_max_guard_sec, 1e-6));
    return clamp01(0.45 * evidence + 0.25 * agreement +
                   0.20 * persistence + 0.10 * compactness);
  }

  void assignEpisode(double detection_time_sec, Decision &decision)
  {
    const bool merge = episode_id_ > 0 &&
        detection_time_sec - last_episode_detection_sec_ <=
            config_.episode_merge_gap_sec &&
        decision.onset_lower_bound_sec <=
            last_episode_upper_bound_sec_ + config_.episode_merge_gap_sec;
    if (merge)
    {
      ++episode_proposal_count_;
      last_episode_lower_bound_sec_ = std::min(
          last_episode_lower_bound_sec_, decision.onset_lower_bound_sec);
      last_episode_upper_bound_sec_ = std::max(
          last_episode_upper_bound_sec_, decision.onset_upper_bound_sec);
    }
    else
    {
      ++episode_id_;
      episode_proposal_count_ = 1;
      last_episode_lower_bound_sec_ = decision.onset_lower_bound_sec;
      last_episode_upper_bound_sec_ = decision.onset_upper_bound_sec;
    }
    last_episode_detection_sec_ = detection_time_sec;
    decision.episode_id = episode_id_;
    decision.proposal_index_in_episode = episode_proposal_count_;
    decision.merged_with_previous_proposal = merge;
    decision.episode_lower_bound_sec = last_episode_lower_bound_sec_;
    decision.episode_upper_bound_sec = last_episode_upper_bound_sec_;
  }

  Config config_;
  std::deque<Sample> baseline_;
  std::deque<Sample> long_history_;
  std::deque<Sample> evidence_history_;
  std::deque<double> sample_intervals_sec_;
  std::size_t consecutive_anomalies_ = 0;
  double candidate_onset_sec_ = -1.0;
  double last_sample_time_sec_ = -1.0;
  bool triggered_ = false;
  bool suppressed_rearm_pending_ = false;
  std::size_t episode_id_ = 0;
  std::size_t episode_proposal_count_ = 0;
  double last_episode_detection_sec_ = -1.0;
  double last_episode_lower_bound_sec_ = -1.0;
  double last_episode_upper_bound_sec_ = -1.0;
};

} // namespace harm_detection
} // namespace fast_livo

#endif // ONLINE_HARM_DETECTOR_H_
