#ifndef FAST_LIVO_CHECKPOINT_BINARY_CODEC_H
#define FAST_LIVO_CHECKPOINT_BINARY_CODEC_H

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fast_livo
{
namespace checkpoint_io
{

using Bytes = std::vector<std::uint8_t>;

inline std::uint64_t fnv1a(const Bytes &bytes)
{
  std::uint64_t value = 1469598103934665603ULL;
  for (const std::uint8_t byte : bytes)
  {
    value ^= byte;
    value *= 1099511628211ULL;
  }
  return value;
}

class Writer
{
public:
  template <typename T> void pod(const T &value)
  {
    static_assert(std::is_trivially_copyable<T>::value,
                  "checkpoint scalar must be trivially copyable");
    const auto *begin = reinterpret_cast<const std::uint8_t *>(&value);
    bytes_.insert(bytes_.end(), begin, begin + sizeof(T));
  }

  void raw(const void *data, const std::size_t size)
  {
    const auto *begin = reinterpret_cast<const std::uint8_t *>(data);
    bytes_.insert(bytes_.end(), begin, begin + size);
  }

  void string(const std::string &value)
  {
    pod(static_cast<std::uint64_t>(value.size()));
    raw(value.data(), value.size());
  }

  template <typename Derived>
  void eigen(const Eigen::MatrixBase<Derived> &value)
  {
    for (Eigen::Index row = 0; row < value.rows(); ++row)
      for (Eigen::Index column = 0; column < value.cols(); ++column)
        pod(value(row, column));
  }

  const Bytes &bytes() const { return bytes_; }
  Bytes take() { return std::move(bytes_); }

private:
  Bytes bytes_;
};

class Reader
{
public:
  explicit Reader(const Bytes &bytes) : bytes_(bytes) {}

  template <typename T> T pod()
  {
    static_assert(std::is_trivially_copyable<T>::value,
                  "checkpoint scalar must be trivially copyable");
    require(sizeof(T));
    T value;
    std::memcpy(&value, bytes_.data() + cursor_, sizeof(T));
    cursor_ += sizeof(T);
    return value;
  }

  void raw(void *destination, const std::size_t size)
  {
    require(size);
    std::memcpy(destination, bytes_.data() + cursor_, size);
    cursor_ += size;
  }

  std::string string()
  {
    const std::uint64_t size = pod<std::uint64_t>();
    if (size > remaining()) throw std::runtime_error("invalid checkpoint string size");
    std::string value(reinterpret_cast<const char *>(bytes_.data() + cursor_),
                      static_cast<std::size_t>(size));
    cursor_ += static_cast<std::size_t>(size);
    return value;
  }

  template <typename Derived> void eigen(Eigen::MatrixBase<Derived> &value)
  {
    for (Eigen::Index row = 0; row < value.rows(); ++row)
      for (Eigen::Index column = 0; column < value.cols(); ++column)
        value(row, column) = pod<typename Derived::Scalar>();
  }

  std::size_t remaining() const { return bytes_.size() - cursor_; }
  std::size_t cursor() const { return cursor_; }
  bool finished() const { return cursor_ == bytes_.size(); }

private:
  void require(const std::size_t size) const
  {
    if (size > remaining())
      throw std::runtime_error(
          "truncated checkpoint payload at byte " + std::to_string(cursor_) +
          " (need " + std::to_string(size) + ", remaining " +
          std::to_string(remaining()) + ")");
  }

  const Bytes &bytes_;
  std::size_t cursor_ = 0;
};

inline std::uint64_t checkedCount(Reader &reader, const std::uint64_t maximum,
                                  const char *label)
{
  const std::uint64_t count = reader.pod<std::uint64_t>();
  if (count > maximum)
    throw std::runtime_error(std::string("invalid checkpoint ") + label + " count");
  return count;
}

} // namespace checkpoint_io
} // namespace fast_livo

#endif
