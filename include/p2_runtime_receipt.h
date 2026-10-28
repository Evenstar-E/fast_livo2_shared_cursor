#ifndef FAST_LIVO_P2_RUNTIME_RECEIPT_H
#define FAST_LIVO_P2_RUNTIME_RECEIPT_H

// Default-off, append-only observation receipt. It is deliberately independent
// of ROS and the mapper checkpoint; restoring an estimator cannot rewind seq.
#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <time.h>

namespace fast_livo { namespace p2_receipt {

struct Row
{
  std::string record_type; // consumed, result_ready, proposal, status
  std::string route;
  std::uint64_t event_id = 0;
  std::string stage;
  std::string modality;
  std::uint64_t sensor_max_input_stamp_ns = 0;
  std::string input_stamp_semantics;
  std::uint64_t proposal_episode_id = 0;
  std::uint64_t proposal_index_in_episode = 0;
  std::string proposal_origin; // natural, frozen_forced, or empty
  std::string onset_lower_sec;
  std::string onset_point_sec;
  std::string onset_upper_sec;
  std::string status;
  std::string detail;
  std::uint64_t buffer_events = 0;
  std::uint64_t replay_consumed_events = 0;
  std::uint64_t replay_total_events = 0;
  std::uint64_t replay_elapsed_ns = 0;
  std::uint64_t rearm_count = 0;
};

inline std::uint64_t monotonicNs()
{
#if defined(__linux__)
  timespec now{};
  if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    throw std::runtime_error("p2 clock_gettime(CLOCK_MONOTONIC) failed");
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
      static_cast<std::uint64_t>(now.tv_nsec);
#else
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

inline std::string clockDomain()
{
#if defined(__linux__)
  std::ifstream boot_id("/proc/sys/kernel/random/boot_id");
  std::string id;
  if (!(boot_id >> id) || id.empty())
    throw std::runtime_error("p2 Linux boot_id unavailable");
  return "linux_boot_id:" + id + ":CLOCK_MONOTONIC";
#else
  return "non_linux_steady_clock_not_cross_process_comparable";
#endif
}

inline std::string csv(const std::string &value)
{
  std::string escaped = "\"";
  for (char character : value)
  {
    if (character == '"') escaped += "\"\"";
    else escaped += character;
  }
  escaped += '"';
  return escaped;
}

class Writer
{
public:
  void open(const std::string &path)
  {
    if (path.empty()) return;
    domain_ = clockDomain();
    output_.open(path, std::ios::out | std::ios::trunc);
    if (!output_)
      throw std::runtime_error("p2 receipt output open failed: " + path);
    output_ << "schema,receipt_seq,clock_domain,steady_ns,record_type,route,event_id,stage,modality,sensor_max_input_stamp_ns,input_stamp_semantics,proposal_episode_id,proposal_index_in_episode,proposal_origin,onset_lower_sec,onset_point_sec,onset_upper_sec,status,detail,buffer_events,replay_consumed_events,replay_total_events,replay_elapsed_ns,rearm_count\n";
    output_.flush();
  }

  bool enabled() const { return output_.is_open(); }

  std::uint64_t write(const Row &row)
  {
    if (!enabled()) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t stamp = monotonicNs();
    const std::uint64_t sequence = ++sequence_;
    output_ << "p2_runtime_receipt_v1," << sequence << ',' << csv(domain_)
            << ',' << stamp << ',' << csv(row.record_type) << ','
            << csv(row.route) << ',' << row.event_id << ','
            << csv(row.stage) << ',' << csv(row.modality) << ','
            << row.sensor_max_input_stamp_ns << ','
            << csv(row.input_stamp_semantics) << ','
            << row.proposal_episode_id << ','
            << row.proposal_index_in_episode << ','
            << csv(row.proposal_origin) << ','
            << csv(row.onset_lower_sec) << ','
            << csv(row.onset_point_sec) << ','
            << csv(row.onset_upper_sec) << ','
            << csv(row.status) << ',' << csv(row.detail) << ','
            << row.buffer_events << ',' << row.replay_consumed_events << ','
            << row.replay_total_events << ',' << row.replay_elapsed_ns << ','
            << row.rearm_count << '\n';
    output_.flush();
    if (!output_)
      throw std::runtime_error("p2 receipt output write failed");
    return sequence;
  }

  std::uint64_t sequence() const { return sequence_; }
  const std::string &domain() const { return domain_; }

private:
  std::ofstream output_;
  std::mutex mutex_;
  std::string domain_;
  std::uint64_t sequence_ = 0;
};

}} // namespace fast_livo::p2_receipt

#endif
