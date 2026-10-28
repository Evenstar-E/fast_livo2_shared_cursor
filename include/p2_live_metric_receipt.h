#ifndef FAST_LIVO_P2_LIVE_METRIC_RECEIPT_H
#define FAST_LIVO_P2_LIVE_METRIC_RECEIPT_H

// Private, default-off observation only. No ROS, mapper state, or public mux.
// This metric is a strict prior-pose-whitened update norm, NOT measurement NIS.
#include "p2_runtime_receipt.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fast_livo { namespace p2_live_metric {

static constexpr const char *kMetricName =
    "pose_prior_whitened_update_norm_strict_v1";
static constexpr const char *kMetricFormula =
    "sqrt(delta_pose6^T * inverse(P_prior_pose6) * delta_pose6)";
static constexpr const char *kMetricUnits = "dimensionless_mixed_pose_whitened";
static constexpr const char *kHeader =
    "schema,receipt_seq,clock_domain,steady_ns,record_type,route,event_id,stage,modality,sensor_max_input_stamp_ns,input_stamp_semantics,checkpoint_sha256,source_prefix_sha256,proposal_id,proposal_episode_id,proposal_index_in_episode,proposal_origin,branch_id,branch_scale,consumed_steady_ns,result_ready_steady_ns,input_clock_domain,source_binding_state,metric_name,metric_formula,metric_units,metric_valid,metric_value,metric_reason,coordinator_eligible,eligibility_reason\n";

struct Metric
{
  bool valid = false;
  double value = 0.0; // serialized empty whenever !valid
  std::string reason = "not_computed_or_no_lio_update";
};

inline Metric computePosePriorWhitenedUpdate(
    const Eigen::Matrix<double, 6, 1> &delta_pose6,
    const Eigen::Matrix<double, 6, 6> &prior_pose_covariance)
{
  Metric result;
  if (!delta_pose6.allFinite())
  {
    result.reason = "nonfinite_pose_update";
    return result;
  }
  if (!prior_pose_covariance.allFinite())
  {
    result.reason = "nonfinite_prior_pose_covariance";
    return result;
  }
  const double scale = prior_pose_covariance.cwiseAbs().maxCoeff();
  const double asymmetry = (prior_pose_covariance -
      prior_pose_covariance.transpose()).cwiseAbs().maxCoeff();
  if (asymmetry > 1e-10 * std::max(1.0, scale))
  {
    result.reason = "asymmetric_prior_pose_covariance";
    return result;
  }
  // Only sub-tolerance roundoff is symmetrized. No diagonal jitter is added;
  // a semidefinite or indefinite covariance is an invalid metric receipt.
  const Eigen::Matrix<double, 6, 6> symmetric =
      0.5 * prior_pose_covariance +
      0.5 * prior_pose_covariance.transpose();
  if (!symmetric.allFinite())
  {
    result.reason = "symmetric_covariance_nonfinite";
    return result;
  }
  Eigen::LLT<Eigen::Matrix<double, 6, 6>> factor(symmetric);
  if (factor.info() != Eigen::Success ||
      !factor.matrixL().toDenseMatrix().allFinite() ||
      (factor.matrixL().toDenseMatrix().diagonal().array() <= 0.0).any())
  {
    result.reason = "prior_pose_covariance_not_spd";
    return result;
  }
  const Eigen::Matrix<double, 6, 1> solved = factor.solve(delta_pose6);
  const double squared = delta_pose6.dot(solved);
  if (!solved.allFinite() || !std::isfinite(squared) || squared < 0.0)
  {
    result.reason = "whitened_square_invalid";
    return result;
  }
  if (squared == 0.0 && delta_pose6.cwiseAbs().maxCoeff() > 0.0)
  {
    result.reason = "whitened_square_underflow";
    return result;
  }
  const double norm = std::sqrt(squared);
  if (!std::isfinite(norm))
  {
    result.reason = "whitened_norm_nonfinite";
    return result;
  }
  result.valid = true;
  result.value = norm;
  result.reason = "valid_strict_spd_no_jitter_no_fallback";
  return result;
}

inline Metric computeObservedPosePriorWhitenedUpdate(
    const Eigen::Matrix<double, 6, 1> &delta_pose6,
    const Eigen::Matrix<double, 6, 6> &prior_pose_covariance,
    std::int64_t effective_observations)
{
  if (effective_observations <= 0)
  {
    Metric result;
    result.reason = "no_effective_lidar_observations";
    return result;
  }
  return computePosePriorWhitenedUpdate(delta_pose6, prior_pose_covariance);
}

struct Row
{
  std::string record_type = "metric_result_ready";
  std::string route;
  std::uint64_t event_id = 0;
  std::string stage = "post_update";
  std::string modality; // LIO, VIO; no-update leaves Metric invalid
  std::uint64_t sensor_max_input_stamp_ns = 0;
  std::string input_stamp_semantics = "consumed_input_availability_upper_bound_only";
  std::string checkpoint_sha256;
  std::string source_prefix_sha256;
  std::string proposal_id;
  std::uint64_t proposal_episode_id = 0;
  std::uint64_t proposal_index_in_episode = 0;
  std::string proposal_origin;
  std::string branch_id;
  double branch_scale = 0.0; // 0 means unavailable, serialized empty
  std::uint64_t consumed_steady_ns = 0;
  std::uint64_t result_ready_steady_ns = 0;
  std::string input_clock_domain;
};

inline bool isSha256(const std::string &digest)
{
  if (digest.size() != 64) return false;
  for (char c : digest)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return false;
  return true;
}

inline std::string decimal(double value)
{
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

inline bool readCsvRecord(std::istream &input, std::string &record)
{
  record.clear();
  bool quoted = false;
  bool started = false;
  char character;
  while (input.get(character))
  {
    started = true;
    if (character == '"')
    {
      record += character;
      if (quoted && input.peek() == '"')
      {
        input.get(character);
        record += character;
      }
      else quoted = !quoted;
    }
    else if (character == '\n' && !quoted)
      return true;
    else record += character;
  }
  if (quoted) throw std::runtime_error("p2 live metric existing CSV quote unclosed");
  if (started) throw std::runtime_error("p2 live metric existing row truncated");
  return false;
}

inline std::size_t csvColumns(const std::string &record)
{
  std::size_t columns = 1;
  bool quoted = false;
  bool field_start = true;
  bool after_quote = false;
  for (std::size_t i = 0; i < record.size(); ++i)
  {
    const char c = record[i];
    if (quoted)
    {
      if (c == '"')
      {
        if (i + 1 < record.size() && record[i + 1] == '"') ++i;
        else { quoted = false; after_quote = true; }
      }
    }
    else if (c == ',')
    {
      ++columns;
      field_start = true;
      after_quote = false;
    }
    else if (c == '"')
    {
      if (!field_start || after_quote)
        throw std::runtime_error("p2 live metric existing CSV quote malformed");
      quoted = true;
      field_start = false;
    }
    else
    {
      if (after_quote)
        throw std::runtime_error("p2 live metric existing CSV field malformed");
      field_start = false;
    }
  }
  if (quoted) throw std::runtime_error("p2 live metric existing CSV quote unclosed");
  return columns;
}

class Writer
{
public:
  void open(const std::string &path)
  {
    if (path.empty()) return; // default-off: no clock read, file, or state
    std::lock_guard<std::mutex> lock(mutex_);
    if (output_.is_open()) throw std::runtime_error("p2 live metric writer already open");
    domain_ = p2_receipt::clockDomain(); // boot-id fail closed on Linux
    std::ifstream existing(path);
    std::string line;
    if (existing && std::getline(existing, line))
    {
      if (existing.eof() || line + "\n" != kHeader)
        throw std::runtime_error("p2 live metric receipt header mismatch");
      while (readCsvRecord(existing, line))
      {
        if (line.empty()) continue;
        if (csvColumns(line) != 31)
          throw std::runtime_error("p2 live metric existing row column mismatch");
        const std::size_t first = line.find(',');
        const std::size_t second = line.find(',', first + 1);
        if (first == std::string::npos || second == std::string::npos ||
            line.substr(0, first) != "p2_live_metric_receipt_v1")
          throw std::runtime_error("p2 live metric existing row corrupt");
        const std::string number = line.substr(first + 1, second - first - 1);
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(number, &consumed);
        if (consumed != number.size() || parsed != sequence_ + 1)
          throw std::runtime_error("p2 live metric existing sequence corrupt");
        sequence_ = static_cast<std::uint64_t>(parsed);
      }
    }
    output_.open(path, std::ios::out | std::ios::app);
    if (!output_) throw std::runtime_error("p2 live metric output open failed");
    if (output_.tellp() == std::streampos(0))
    {
      output_ << kHeader;
      output_.flush();
      if (!output_) throw std::runtime_error("p2 live metric header write failed");
    }
  }

  bool enabled() const { return output_.is_open(); }
  std::uint64_t sequence() const { return sequence_; }
  const std::string &domain() const { return domain_; }

  std::uint64_t write(const Row &row, const Metric &metric)
  {
    if (!enabled()) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t now = p2_receipt::monotonicNs();
    const bool binding_declared = isSha256(row.checkpoint_sha256) &&
        isSha256(row.source_prefix_sha256) && !row.proposal_id.empty() &&
        row.proposal_episode_id != 0 && row.proposal_index_in_episode != 0 &&
        !row.proposal_origin.empty() && !row.branch_id.empty() &&
        std::isfinite(row.branch_scale) && row.branch_scale > 0.0;
    const bool timing_declared = row.consumed_steady_ns != 0 &&
        row.result_ready_steady_ns != 0 &&
        row.consumed_steady_ns <= row.result_ready_steady_ns &&
        row.result_ready_steady_ns <= now && row.input_clock_domain == domain_;
    const bool metric_valid = metric.valid && std::isfinite(metric.value) &&
        metric.value >= 0.0;
    // This writer verifies neither source bytes nor common geometric support.
    // Even fully declared metadata remains private and coordinator-ineligible.
    const std::string binding = binding_declared ?
        "declared_unverified" : "unbound";
    const std::string eligibility = !binding_declared ?
        "source_binding_missing_or_malformed" :
        !timing_declared ? "timing_missing_or_invalid" :
        !metric_valid ? "metric_invalid" :
        "source_content_and_common_support_not_verified";
    const std::uint64_t seq = sequence_ + 1;
    output_ << "p2_live_metric_receipt_v1," << seq << ','
            << p2_receipt::csv(domain_) << ',' << now << ','
            << p2_receipt::csv(row.record_type) << ','
            << p2_receipt::csv(row.route) << ',' << row.event_id << ','
            << p2_receipt::csv(row.stage) << ','
            << p2_receipt::csv(row.modality) << ','
            << row.sensor_max_input_stamp_ns << ','
            << p2_receipt::csv(row.input_stamp_semantics) << ','
            << p2_receipt::csv(row.checkpoint_sha256) << ','
            << p2_receipt::csv(row.source_prefix_sha256) << ','
            << p2_receipt::csv(row.proposal_id) << ','
            << row.proposal_episode_id << ',' << row.proposal_index_in_episode
            << ',' << p2_receipt::csv(row.proposal_origin) << ','
            << p2_receipt::csv(row.branch_id) << ','
            << (std::isfinite(row.branch_scale) && row.branch_scale > 0.0 ?
                decimal(row.branch_scale) : std::string()) << ','
            << row.consumed_steady_ns << ',' << row.result_ready_steady_ns << ','
            << p2_receipt::csv(row.input_clock_domain) << ','
            << p2_receipt::csv(binding) << ','
            << p2_receipt::csv(kMetricName) << ','
            << p2_receipt::csv(kMetricFormula) << ','
            << p2_receipt::csv(kMetricUnits) << ','
            << (metric_valid ? 1 : 0) << ','
            << (metric_valid ? decimal(metric.value) : std::string()) << ','
            << p2_receipt::csv(metric_valid ? metric.reason :
                (metric.valid ? "metric_value_invalid" : metric.reason)) << ','
            << 0 << ',' << p2_receipt::csv(eligibility) << '\n';
    output_.flush();
    if (!output_) throw std::runtime_error("p2 live metric receipt write failed");
    sequence_ = seq;
    return seq;
  }

private:
  mutable std::mutex mutex_;
  std::ofstream output_;
  std::string domain_;
  std::uint64_t sequence_ = 0;
};

}} // namespace fast_livo::p2_live_metric

#endif
