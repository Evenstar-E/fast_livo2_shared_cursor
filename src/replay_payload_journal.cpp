#include "replay_payload_journal.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace
{
using Bytes = std::vector<unsigned char>;
constexpr std::uint64_t kRecordMagic = 0x32574152504c5246ULL; // FRLPRAW2
constexpr std::uint32_t kRecordSchema = 2;

template <typename T>
void appendPod(Bytes &bytes, const T &value)
{
  static_assert(std::is_trivially_copyable<T>::value,
                "journal POD must be trivially copyable");
  const auto *begin = reinterpret_cast<const unsigned char *>(&value);
  bytes.insert(bytes.end(), begin, begin + sizeof(T));
}

template <typename T>
T readPod(const Bytes &bytes, std::size_t &cursor)
{
  static_assert(std::is_trivially_copyable<T>::value,
                "journal POD must be trivially copyable");
  if (cursor + sizeof(T) > bytes.size())
    throw std::runtime_error("truncated replay payload journal record");
  T value;
  std::memcpy(&value, bytes.data() + cursor, sizeof(T));
  cursor += sizeof(T);
  return value;
}

void appendCloud(Bytes &bytes, const PointCloudXYZI::Ptr &cloud)
{
  const std::uint64_t count = cloud == nullptr ? 0 : cloud->points.size();
  appendPod(bytes, count);
  const std::uint32_t width = cloud == nullptr ? 0 : cloud->width;
  const std::uint32_t height = cloud == nullptr ? 0 : cloud->height;
  const std::uint8_t dense = cloud != nullptr && cloud->is_dense ? 1 : 0;
  appendPod(bytes, width);
  appendPod(bytes, height);
  appendPod(bytes, dense);
  if (count == 0) return;
  const auto *begin = reinterpret_cast<const unsigned char *>(cloud->points.data());
  const std::size_t cloud_begin = bytes.size();
  bytes.insert(bytes.end(), begin, begin + count * sizeof(PointType));
  // PCL PointXYZINormal leaves data_c[2:4] uninitialized. These slots are not
  // registered point fields, but serializing the raw struct would persist
  // process-dependent bytes and make equivalent journals hash differently.
  // Preserve x/y/z, homogeneous coordinates, normals, intensity and curvature.
  const auto *padding = reinterpret_cast<const unsigned char *>(
      &cloud->points.front().data_c[2]);
  const std::size_t padding_offset = padding - begin;
  if (padding_offset + 2 * sizeof(float) > sizeof(PointType))
    throw std::logic_error("unexpected PointXYZINormal padding layout");
  for (std::uint64_t index = 0; index < count; ++index)
    std::memset(bytes.data() + cloud_begin + index * sizeof(PointType) +
                    padding_offset, 0, 2 * sizeof(float));
}

PointCloudXYZI::Ptr readCloud(const Bytes &bytes, std::size_t &cursor)
{
  const std::uint64_t count = readPod<std::uint64_t>(bytes, cursor);
  const std::uint32_t width = readPod<std::uint32_t>(bytes, cursor);
  const std::uint32_t height = readPod<std::uint32_t>(bytes, cursor);
  const std::uint8_t dense = readPod<std::uint8_t>(bytes, cursor);
  if (count > (bytes.size() - cursor) / sizeof(PointType))
    throw std::runtime_error("invalid point count in replay payload journal");
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
  cloud->points.resize(static_cast<std::size_t>(count));
  if (count != 0)
  {
    std::memcpy(cloud->points.data(), bytes.data() + cursor,
                static_cast<std::size_t>(count) * sizeof(PointType));
    cursor += static_cast<std::size_t>(count) * sizeof(PointType);
  }
  cloud->width = width;
  cloud->height = height;
  cloud->is_dense = dense != 0;
  return cloud;
}

void appendRawFrame(Bytes &bytes, const RawLidarFrame &frame)
{
  appendPod(bytes, static_cast<std::uint8_t>(frame.message_type));
  appendPod(bytes, frame.sensor_header_time);
  appendPod(bytes, frame.effective_header_time);
  appendPod(bytes, frame.minimum_processed_offset_ms);
  appendPod(bytes, frame.maximum_processed_offset_ms);
  appendPod(bytes, frame.oracle_source_relative_time);
  appendPod(bytes, frame.oracle_inverse_range_scale);
  appendPod(bytes, static_cast<std::uint8_t>(
      frame.oracle_range_scale_active ? 1 : 0));
  appendPod(bytes, frame.serialized_checksum);
  appendPod(bytes, static_cast<std::uint64_t>(frame.serialized_message.size()));
  bytes.insert(bytes.end(), frame.serialized_message.begin(),
               frame.serialized_message.end());
}

RawLidarFrame readRawFrame(const Bytes &bytes, std::size_t &cursor)
{
  RawLidarFrame frame;
  const std::uint8_t type = readPod<std::uint8_t>(bytes, cursor);
  if (type > static_cast<std::uint8_t>(RawLidarMessageType::LIVOX_CUSTOM))
    throw std::runtime_error("invalid raw LiDAR message type in journal");
  frame.message_type = static_cast<RawLidarMessageType>(type);
  frame.sensor_header_time = readPod<double>(bytes, cursor);
  frame.effective_header_time = readPod<double>(bytes, cursor);
  frame.minimum_processed_offset_ms = readPod<float>(bytes, cursor);
  frame.maximum_processed_offset_ms = readPod<float>(bytes, cursor);
  frame.oracle_source_relative_time = readPod<double>(bytes, cursor);
  frame.oracle_inverse_range_scale = readPod<double>(bytes, cursor);
  frame.oracle_range_scale_active =
      readPod<std::uint8_t>(bytes, cursor) != 0;
  frame.serialized_checksum = readPod<std::uint64_t>(bytes, cursor);
  const std::uint64_t size = readPod<std::uint64_t>(bytes, cursor);
  if (size > bytes.size() - cursor)
    throw std::runtime_error("invalid raw LiDAR payload size in journal");
  frame.serialized_message.assign(bytes.begin() + cursor,
                                  bytes.begin() + cursor + size);
  cursor += static_cast<std::size_t>(size);
  return frame;
}

void appendImage(Bytes &bytes, const cv::Mat &image)
{
  const std::int32_t rows = image.rows;
  const std::int32_t cols = image.cols;
  const std::int32_t type = image.empty() ? 0 : image.type();
  const std::uint64_t row_bytes = image.empty() ? 0 :
      static_cast<std::uint64_t>(image.cols * image.elemSize());
  appendPod(bytes, rows);
  appendPod(bytes, cols);
  appendPod(bytes, type);
  appendPod(bytes, row_bytes);
  for (int row = 0; row < rows; ++row)
  {
    const unsigned char *begin = image.ptr<unsigned char>(row);
    bytes.insert(bytes.end(), begin, begin + row_bytes);
  }
}

cv::Mat readImage(const Bytes &bytes, std::size_t &cursor)
{
  const std::int32_t rows = readPod<std::int32_t>(bytes, cursor);
  const std::int32_t cols = readPod<std::int32_t>(bytes, cursor);
  const std::int32_t type = readPod<std::int32_t>(bytes, cursor);
  const std::uint64_t row_bytes = readPod<std::uint64_t>(bytes, cursor);
  if (rows == 0 || cols == 0) return cv::Mat();
  if (rows < 0 || cols < 0)
    throw std::runtime_error("invalid image shape in replay payload journal");
  cv::Mat image(rows, cols, type);
  const std::uint64_t expected = static_cast<std::uint64_t>(cols * image.elemSize());
  if (row_bytes != expected ||
      static_cast<std::uint64_t>(rows) * row_bytes > bytes.size() - cursor)
    throw std::runtime_error("invalid image payload in replay journal");
  for (int row = 0; row < rows; ++row)
  {
    std::memcpy(image.ptr<unsigned char>(row), bytes.data() + cursor,
                static_cast<std::size_t>(row_bytes));
    cursor += static_cast<std::size_t>(row_bytes);
  }
  return image;
}

std::uint64_t fnv1a(const Bytes &bytes)
{
  std::uint64_t value = 1469598103934665603ULL;
  for (const unsigned char byte : bytes)
  {
    value ^= byte;
    value *= 1099511628211ULL;
  }
  return value;
}
} // namespace

namespace fast_livo
{
namespace replay_storage
{

Journal::~Journal()
{
  // Destructors cannot report I/O failure; explicit commitDurable() is the
  // checked persistence boundary used before replay.
  stopWriter();
}

void Journal::configureAsync(const bool enabled,
                             const std::uint64_t maximum_queued_bytes)
{
  if (stream_.is_open() || writer_.joinable())
    throw std::logic_error("configure replay journal before opening it");
  if (enabled && maximum_queued_bytes == 0)
    throw std::invalid_argument("async replay journal queue limit must be positive");
  async_enabled_ = enabled;
  maximum_queued_bytes_ = maximum_queued_bytes;
}

void Journal::rethrowWriterError() const
{
  if (writer_error_) std::rethrow_exception(writer_error_);
}

void Journal::stopWriter()
{
  if (!writer_.joinable()) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  changed_.notify_all();
  writer_.join();
}

void Journal::writerLoop()
{
  try
  {
    for (;;)
    {
      PendingRecord pending;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] {
          return stopping_ || !pending_.empty();
        });
        if (pending_.empty()) return;
        pending = std::move(pending_.front());
        pending_.pop_front();
      }
      stream_.clear();
      stream_.seekp(static_cast<std::streamoff>(pending.offset), std::ios::beg);
      stream_.write(reinterpret_cast<const char *>(pending.bytes.data()),
                    static_cast<std::streamsize>(pending.bytes.size()));
      stream_.flush();
      if (!stream_)
        throw std::runtime_error("cannot write async replay payload journal");
      const std::uint64_t completed_size = pending.bytes.size();
      Bytes().swap(pending.bytes);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        visible_bytes_ = pending.offset + completed_size;
        queued_bytes_ -= completed_size;
      }
      changed_.notify_all();
    }
  }
  catch (...)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      writer_error_ = std::current_exception();
    }
    changed_.notify_all();
  }
}

void Journal::waitVisible(const std::uint64_t end_offset)
{
  if (!async_enabled_) return;
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [this, end_offset] {
    return visible_bytes_ >= end_offset || writer_error_;
  });
  rethrowWriterError();
}

void Journal::commitDurable()
{
  if (!enabled() || !stream_.is_open()) return;
  const std::uint64_t target = bytes_written_;
  waitVisible(target);
  if (!async_enabled_)
  {
    stream_.flush();
    if (!stream_) throw std::runtime_error("cannot flush replay payload journal");
  }
  if (target <= durable_bytes_) return;
  const int fd = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("cannot open replay payload journal for fsync");
  const int result = ::fsync(fd);
  ::close(fd);
  if (result != 0) throw std::runtime_error("cannot fsync replay payload journal");
  durable_bytes_ = target;
}

void Journal::ensureOpen()
{
  if (!enabled()) throw std::logic_error("replay payload journal path is empty");
  if (stream_.is_open()) return;
  stream_.open(path_.c_str(), std::ios::in | std::ios::out |
                              std::ios::binary | std::ios::trunc);
  if (!stream_.is_open())
    throw std::runtime_error("cannot open replay payload journal: " + path_);
  if (async_enabled_)
  {
    stopping_ = false;
    writer_error_ = nullptr;
    writer_ = std::thread(&Journal::writerLoop, this);
  }
}

void Journal::reset()
{
  if (!enabled()) return;
  // A reset discards the prior generation; wait for the writer to finish so
  // truncation cannot race it, but do not fsync bytes about to be discarded.
  if (stream_.is_open()) waitVisible(bytes_written_);
  stopWriter();
  if (read_stream_.is_open()) read_stream_.close();
  if (stream_.is_open()) stream_.close();
  bytes_written_ = 0;
  visible_bytes_ = 0;
  durable_bytes_ = 0;
  queued_bytes_ = 0;
  peak_queued_bytes_ = 0;
  backpressure_wall_sec_ = 0.0;
  pending_.clear();
  records_written_ = 0;
  records_read_ = 0;
  read_wall_sec_ = 0.0;
  ensureOpen();
}

Record Journal::spill(LidarMeasureGroup &group)
{
  ensureOpen();
  Bytes bytes;
  appendPod(bytes, kRecordMagic);
  appendPod(bytes, kRecordSchema);
  appendCloud(bytes, group.lidar);
  appendCloud(bytes, group.pcl_proc_cur);
  appendCloud(bytes, group.pcl_proc_next);
  appendPod(bytes, group.raw_reprocess_begin_time);
  appendPod(bytes, group.raw_reprocess_end_time);
  appendPod(bytes, static_cast<std::uint8_t>(
      group.raw_reprocess_recipe_valid ? 1 : 0));
  appendPod(bytes, static_cast<std::uint8_t>(
      group.raw_reprocess_split_by_time ? 1 : 0));
  appendPod(bytes, group.raw_preprocess_lidar_type);
  appendPod(bytes, group.raw_preprocess_point_filter_num);
  appendPod(bytes, group.raw_preprocess_scan_lines);
  appendPod(bytes, group.raw_preprocess_blind);
  appendPod(bytes, static_cast<std::uint8_t>(
      group.raw_preprocess_feature_enabled ? 1 : 0));
  appendPod(bytes, static_cast<std::uint64_t>(group.raw_lidar_frames.size()));
  for (const RawLidarFrame &frame : group.raw_lidar_frames)
    appendRawFrame(bytes, frame);
  appendPod(bytes, static_cast<std::uint64_t>(group.measures.size()));
  for (const MeasureGroup &measure : group.measures) appendImage(bytes, measure.img);

  Record record;
  record.offset = bytes_written_;
  record.size = bytes.size();
  record.checksum = fnv1a(bytes);
  record.spilled = true;
  if (async_enabled_)
  {
    if (record.size > maximum_queued_bytes_)
      throw std::runtime_error("replay payload record exceeds async queue limit");
    const auto wait_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this, &record] {
      return queued_bytes_ + record.size <= maximum_queued_bytes_ || writer_error_;
    });
    rethrowWriterError();
    backpressure_wall_sec_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wait_start).count();
    queued_bytes_ += record.size;
    peak_queued_bytes_ = std::max(peak_queued_bytes_, queued_bytes_);
    pending_.push_back(PendingRecord{record.offset, std::move(bytes)});
    lock.unlock();
    changed_.notify_all();
  }
  else
  {
    stream_.clear();
    stream_.seekp(0, std::ios::end);
    const std::streampos position = stream_.tellp();
    if (position < 0 || static_cast<std::uint64_t>(position) != record.offset)
      throw std::runtime_error("cannot seek replay payload journal");
    stream_.write(reinterpret_cast<const char *>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    stream_.flush();
    if (!stream_) throw std::runtime_error("cannot write replay payload journal");
    visible_bytes_ += record.size;
  }
  bytes_written_ += record.size;
  ++records_written_;

  group.lidar.reset(new PointCloudXYZI());
  group.pcl_proc_cur.reset(new PointCloudXYZI());
  group.pcl_proc_next.reset(new PointCloudXYZI());
  group.raw_lidar_frames.clear();
  for (MeasureGroup &measure : group.measures) measure.img.release();
  return record;
}

LidarMeasureGroup Journal::materialize(const LidarMeasureGroup &metadata,
                                       const Record &record)
{
  if (!record.spilled) return metadata;
  ensureOpen();
  const auto start = std::chrono::steady_clock::now();
  Bytes bytes(static_cast<std::size_t>(record.size));
  waitVisible(record.offset + record.size);
  std::istream *input = &stream_;
  if (async_enabled_)
  {
    if (!read_stream_.is_open())
      read_stream_.open(path_.c_str(), std::ios::in | std::ios::binary);
    input = &read_stream_;
  }
  input->clear();
  input->seekg(static_cast<std::streamoff>(record.offset), std::ios::beg);
  input->read(reinterpret_cast<char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!*input) throw std::runtime_error("cannot read replay payload journal");
  if (fnv1a(bytes) != record.checksum)
    throw std::runtime_error("replay payload journal checksum mismatch");

  LidarMeasureGroup group = metadata;
  std::size_t cursor = 0;
  const std::uint64_t magic = readPod<std::uint64_t>(bytes, cursor);
  const std::uint32_t schema = readPod<std::uint32_t>(bytes, cursor);
  if (magic != kRecordMagic || schema != kRecordSchema)
    throw std::runtime_error("unsupported replay payload journal schema");
  group.lidar = readCloud(bytes, cursor);
  group.pcl_proc_cur = readCloud(bytes, cursor);
  group.pcl_proc_next = readCloud(bytes, cursor);
  group.raw_reprocess_begin_time = readPod<double>(bytes, cursor);
  group.raw_reprocess_end_time = readPod<double>(bytes, cursor);
  group.raw_reprocess_recipe_valid =
      readPod<std::uint8_t>(bytes, cursor) != 0;
  group.raw_reprocess_split_by_time =
      readPod<std::uint8_t>(bytes, cursor) != 0;
  group.raw_preprocess_lidar_type = readPod<int>(bytes, cursor);
  group.raw_preprocess_point_filter_num = readPod<int>(bytes, cursor);
  group.raw_preprocess_scan_lines = readPod<int>(bytes, cursor);
  group.raw_preprocess_blind = readPod<double>(bytes, cursor);
  group.raw_preprocess_feature_enabled =
      readPod<std::uint8_t>(bytes, cursor) != 0;
  const std::uint64_t raw_frame_count = readPod<std::uint64_t>(bytes, cursor);
  if (raw_frame_count > bytes.size())
    throw std::runtime_error("invalid raw LiDAR frame count in journal");
  group.raw_lidar_frames.clear();
  group.raw_lidar_frames.reserve(static_cast<std::size_t>(raw_frame_count));
  for (std::uint64_t i = 0; i < raw_frame_count; ++i)
    group.raw_lidar_frames.push_back(readRawFrame(bytes, cursor));
  const std::uint64_t image_count = readPod<std::uint64_t>(bytes, cursor);
  if (image_count != group.measures.size())
    throw std::runtime_error("replay payload journal measure-count mismatch");
  for (MeasureGroup &measure : group.measures) measure.img = readImage(bytes, cursor);
  if (cursor != bytes.size())
    throw std::runtime_error("trailing bytes in replay payload journal record");

  ++records_read_;
  read_wall_sec_ += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  return group;
}

} // namespace replay_storage
} // namespace fast_livo
