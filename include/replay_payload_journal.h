#ifndef FAST_LIVO_REPLAY_PAYLOAD_JOURNAL_H
#define FAST_LIVO_REPLAY_PAYLOAD_JOURNAL_H

#include "common_lib.h"
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <exception>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fast_livo
{
namespace replay_storage
{

struct Record
{
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  std::uint64_t checksum = 0;
  bool spilled = false;
};

// Append-only storage for the heavy, immutable portion of a replay event.
// Scalar timing and shared immutable IMU messages stay in the event descriptor;
// point clouds and images are materialized only while an event is replayed.
class Journal
{
public:
  Journal() = default;
  explicit Journal(const std::string &path) : path_(path) {}
  ~Journal();
  Journal(const Journal &) = delete;
  Journal &operator=(const Journal &) = delete;

  void setPath(const std::string &path) { path_ = path; }
  // Configure before reset/spill. The queue is byte-bounded, including the
  // record currently owned by the writer. Oversize records fail explicitly.
  void configureAsync(bool enabled, std::uint64_t maximum_queued_bytes);
  const std::string &path() const { return path_; }
  bool enabled() const { return !path_.empty(); }
  bool asyncEnabled() const { return async_enabled_; }

  void reset();
  Record spill(LidarMeasureGroup &group);
  LidarMeasureGroup materialize(const LidarMeasureGroup &metadata,
                                const Record &record);
  // A replay/published-state boundary: all accepted records are flushed and
  // fsync'ed before this returns. Reads only need the weaker visibility wait.
  void commitDurable();

  std::uint64_t bytesWritten() const { return bytes_written_; }
  std::uint64_t recordsWritten() const { return records_written_; }
  std::uint64_t recordsRead() const { return records_read_; }
  double readWallSec() const { return read_wall_sec_; }
  std::uint64_t durableBytes() const { return durable_bytes_; }
  std::uint64_t peakQueuedBytes() const { return peak_queued_bytes_; }
  double backpressureWallSec() const { return backpressure_wall_sec_; }

private:
  void ensureOpen();
  void stopWriter();
  void writerLoop();
  void waitVisible(std::uint64_t end_offset);
  void rethrowWriterError() const;

  struct PendingRecord
  {
    std::uint64_t offset = 0;
    std::vector<unsigned char> bytes;
  };

  std::string path_;
  std::fstream stream_;
  std::ifstream read_stream_;
  bool async_enabled_ = false;
  std::uint64_t maximum_queued_bytes_ = 0;
  std::thread writer_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<PendingRecord> pending_;
  bool stopping_ = false;
  std::exception_ptr writer_error_;
  std::uint64_t queued_bytes_ = 0;
  std::uint64_t visible_bytes_ = 0;
  std::uint64_t durable_bytes_ = 0;
  std::uint64_t peak_queued_bytes_ = 0;
  double backpressure_wall_sec_ = 0.0;
  std::uint64_t bytes_written_ = 0;
  std::uint64_t records_written_ = 0;
  std::uint64_t records_read_ = 0;
  double read_wall_sec_ = 0.0;
};

} // namespace replay_storage
} // namespace fast_livo

#endif
