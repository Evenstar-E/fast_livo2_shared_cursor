#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <time.h>

namespace fast_livo { namespace audit {

// Opt-in diagnostic. Process CPU includes OpenMP workers and other mapper
// threads; it is not the CPU time of this callback alone. Each row is flushed
// because the mapper intentionally terminates with _Exit after saving results.
class InputEventProfile {
public:
  void open(const std::string &path) {
    if (path.empty()) return;
    stream_.open(path, std::ios::out | std::ios::trunc);
    if (!stream_) throw std::runtime_error("Cannot open input event profile: " + path);
    stream_ << "kind,sequence,header_seq,stamp_ns,payload_bytes,relative_time_sec,disposition,wall_sec,process_cpu_sec\n";
    stream_ << std::setprecision(17);
    stream_.flush();
  }

  bool enabled() const { return stream_.is_open(); }

  class Scope {
  public:
    Scope(InputEventProfile &profile, const char *kind, std::uint64_t sequence,
          std::uint32_t header_seq, std::uint64_t stamp_ns,
          std::uint64_t payload_bytes, double relative_time_sec = -1.0)
        : profile_(profile), kind_(kind), sequence_(sequence),
          header_seq_(header_seq), stamp_ns_(stamp_ns),
          payload_bytes_(payload_bytes), relative_time_sec_(relative_time_sec) {
      if (profile_.enabled()) {
        clock_gettime(CLOCK_MONOTONIC, &wall_start_);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start_);
      }
    }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    void disposition(const char *value) { disposition_ = value; }

    ~Scope() {
      if (!profile_.enabled()) return;
      timespec wall_end{}, cpu_end{};
      clock_gettime(CLOCK_MONOTONIC, &wall_end);
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
      profile_.stream_ << kind_ << ',' << sequence_ << ',' << header_seq_
          << ',' << stamp_ns_ << ',' << payload_bytes_ << ','
          << relative_time_sec_ << ',' << disposition_ << ','
          << elapsed(wall_start_, wall_end) << ','
          << elapsed(cpu_start_, cpu_end) << '\n';
      profile_.stream_.flush();
    }

  private:
    static double elapsed(const timespec &start, const timespec &end) {
      return static_cast<double>(end.tv_sec - start.tv_sec) +
          static_cast<double>(end.tv_nsec - start.tv_nsec) * 1e-9;
    }
    InputEventProfile &profile_;
    const char *kind_;
    std::uint64_t sequence_;
    std::uint32_t header_seq_;
    std::uint64_t stamp_ns_;
    std::uint64_t payload_bytes_;
    double relative_time_sec_;
    const char *disposition_ = "unclassified";
    timespec wall_start_{};
    timespec cpu_start_{};
  };

private:
  std::ofstream stream_;
};

}} // namespace fast_livo::audit
