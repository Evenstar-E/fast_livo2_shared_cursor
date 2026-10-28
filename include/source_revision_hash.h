#pragma once

#include <openssl/evp.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace fast_livo::source_revision
{
inline bool validSha256(const std::string &value)
{
  if (value.size() != 64) return false;
  for (char ch : value)
    if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
      return false;
  return true;
}

inline std::string sha256File(const std::filesystem::path &path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open source revision payload: " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("source revision SHA-256 initialization failed");
  std::array<char, 1 << 20> buffer{};
  while (input)
  {
    input.read(buffer.data(), buffer.size());
    const auto count = input.gcount();
    if (count > 0 && EVP_DigestUpdate(ctx.get(), buffer.data(),
                                      static_cast<std::size_t>(count)) != 1)
      throw std::runtime_error("source revision SHA-256 update failed");
  }
  if (!input.eof())
    throw std::runtime_error("source revision payload read failed: " + path.string());
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(ctx.get(), digest.data(), &length) != 1 || length != 32)
    throw std::runtime_error("source revision SHA-256 finalization failed");
  constexpr char alphabet[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(64);
  for (unsigned int index = 0; index < length; ++index)
  {
    hex += alphabet[digest[index] >> 4];
    hex += alphabet[digest[index] & 15];
  }
  return hex;
}

inline void requireSha256(const std::filesystem::path &path,
                          const std::string &expected)
{
  if (!validSha256(expected) || sha256File(path) != expected)
    throw std::runtime_error("source revision SHA-256 mismatch: " + path.string());
}
}  // namespace fast_livo::source_revision
