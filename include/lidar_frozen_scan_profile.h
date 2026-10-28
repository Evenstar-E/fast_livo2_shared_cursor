#pragma once

// Private input operator only. Loading a file does NOT attest estimator health,
// causal selection, or authorize publication. Preflight precedes map restoration.
#include "source_revision_hash.h"
#include <openssl/evp.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace fast_livo::scan_profile
{
inline void require(bool condition, const std::string &reason)
{
  if (!condition) throw std::runtime_error("scan_profile:" + reason);
}
inline bool hashValid(const std::string &s)
{
  return source_revision::validSha256(s) && s != std::string(64, '0');
}
inline std::string sha256Bytes(const std::uint8_t *bytes, std::size_t size)
{
  unsigned char output[EVP_MAX_MD_SIZE]; unsigned int length = 0;
  require(EVP_Digest(bytes, size, output, &length, EVP_sha256(), nullptr) == 1 &&
          length == 32, "sha256_failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned int i = 0; i < length; ++i)
  { result += hex[output[i] >> 4]; result += hex[output[i] & 15]; }
  return result;
}
inline std::uint64_t integer(const std::string &s)
{
  require(!s.empty() && (s.size() == 1 || s[0] != '0'), "integer_syntax");
  std::uint64_t value = 0;
  for (char c : s)
  {
    require(c >= '0' && c <= '9', "integer_syntax");
    const unsigned digit = c - '0';
    require(value <= (std::numeric_limits<std::uint64_t>::max() - digit) / 10,
            "integer_overflow");
    value = value * 10 + digit;
  }
  return value;
}
struct Context
{
  std::string checkpoint_sha256, journal_sha256, input_prefix_sha256;
  std::uint64_t checkpoint_ns = 0, decision_event = 0, decision_ns = 0;
  std::uint64_t episode = 0, proposal = 0;
};
struct Scan
{
  std::uint64_t ordinal = 0, header_ns = 0;
  std::string raw_sha256, status, support_sha256;
  double inverse_scale = 1.0;
};
class FrozenProfile
{
public:
  static constexpr std::size_t maximum_scans = 256, maximum_calls = 6000;
  // Tab-delimited wire representation compiled from the existing JSON capsule.
  // Bounded input, fixed field counts, canonical integer spelling; no defaults.
  void load(const std::string &path)
  {
    require(!loaded_, "already_loaded");
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "file_missing");
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    require(size > 0 && size <= 1048576, "file_budget");
    input.seekg(0);
    const auto line = [&input]() {
      std::string s; require(static_cast<bool>(std::getline(input, s)), "truncated_file");
      require(s.size() <= 4096 && s.find('\r') == std::string::npos, "line_syntax");
      std::vector<std::string> fields; std::istringstream stream(s);
      for (std::string f; std::getline(stream, f, '\t');) fields.push_back(f);
      return fields;
    };
    auto fields = line();
    require(fields == std::vector<std::string>{"lidar_frozen_scan_profile_wire_v1",
            "private_input_transport", "public_switch_authorized=false"}, "scope");
    fields = line(); require(fields.size() == 9 && fields[0] == "context", "context_fields");
    Context ctx{fields[1], fields[2], fields[3], integer(fields[4]),
                integer(fields[5]), integer(fields[6]), integer(fields[7]), integer(fields[8])};
    require(hashValid(ctx.checkpoint_sha256) && hashValid(ctx.journal_sha256) &&
            hashValid(ctx.input_prefix_sha256), "context_hash");
    require(ctx.checkpoint_ns > 0 && ctx.decision_event > 0 && ctx.episode > 0 &&
            ctx.proposal > 0 && ctx.checkpoint_ns < ctx.decision_ns, "context_time");
    fields = line(); require(fields.size() == 2 && fields[0] == "method" &&
            !fields[1].empty(), "method");
    const std::string method = fields[1];
    fields = line(); require(fields.size() == 2 && fields[0] == "bindings" &&
            hashValid(fields[1]), "bindings_hash");
    const std::string expected_bindings = fields[1];
    fields = line(); require(fields.size() == 2 && fields[0] == "scans", "scan_count");
    const std::size_t count = integer(fields[1]);
    require(count > 0 && count <= maximum_scans, "scan_budget");
    std::map<std::uint64_t, Scan> scans;
    std::string binding_bytes;
    std::uint64_t previous = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
      fields = line(); require(fields.size() == 8 && fields[0] == "scan", "scan_fields");
      std::size_t parsed = 0; const double q = std::stod(fields[4], &parsed);
      require(parsed == fields[4].size() && std::isfinite(q) && q >= .9 && q <= 1.1,
              "scale_bound");
      Scan scan{integer(fields[1]), integer(fields[2]), fields[3], fields[5], fields[6], q};
      require(fields[7] == "original_raw_once", "operator_scope");
      require(scan.ordinal > 0 && (i == 0 || scan.ordinal > previous) && scan.header_ns > 0 &&
              scan.header_ns <= ctx.decision_ns && hashValid(scan.raw_sha256), "scan_binding");
      require(scan.status == "identity" || scan.status == "supported" ||
              scan.status == "unknown_identity_carry", "scan_status");
      require(q == 1.0 || (scan.status == "supported" && hashValid(scan.support_sha256)),
              "unsupported_correction");
      require(scan.header_ns > ctx.checkpoint_ns || q == 1.0, "precheckpoint_modified");
      require(scans.emplace(scan.header_ns, scan).second, "duplicate_header");
      binding_bytes += fields[1] + "\t" + fields[2] + "\t" + fields[3] + "\n";
      previous = scan.ordinal;
    }
    require(input.peek() == std::char_traits<char>::eof(), "extra_records");
    require(sha256Bytes(reinterpret_cast<const std::uint8_t *>(binding_bytes.data()),
                        binding_bytes.size()) == expected_bindings, "bindings_mismatch");
    context_ = std::move(ctx); scans_ = std::move(scans); method_ = method;
    loaded_ = true;
  }
  void beginPreflight(const Context &actual)
  {
    require(loaded_ && !admitted_ && !preflight_started_, "preflight_state");
    require(actual.checkpoint_sha256 == context_.checkpoint_sha256 &&
            actual.journal_sha256 == context_.journal_sha256 &&
            actual.checkpoint_ns == context_.checkpoint_ns &&
            actual.decision_event == context_.decision_event &&
            actual.decision_ns == context_.decision_ns && actual.episode == context_.episode &&
            actual.proposal == context_.proposal, "runtime_context_mismatch");
    // input_prefix is the estimator's declared support binding, not a runtime
    // causal certificate. The actual full replay input is bound by journal SHA.
    preflight_started_ = true;
  }
  void observe(std::uint64_t header_ns, const std::vector<std::uint8_t> &original,
               std::uint64_t event_ns)
  {
    try
    {
      require(preflight_started_ && !admitted_ && !failed_, "observe_state");
      require(++preflight_calls_ <= maximum_calls, "preflight_call_budget");
      const Scan &scan = verify(header_ns, original, event_ns);
      observed_.insert(scan.header_ns);
    }
    catch (...) { failed_ = true; throw; }
  }
  void finishPreflight()
  {
    require(preflight_started_ && !admitted_ && !failed_ && observed_.size() == scans_.size(),
            "incomplete_replay_binding_set");
    admitted_ = true;
  }
  double consume(std::uint64_t header_ns, const std::vector<std::uint8_t> &original,
                 std::uint64_t event_ns)
  {
    require(admitted_, "not_admitted");
    require(++calls_ <= maximum_calls, "call_budget");
    return verify(header_ns, original, event_ns).inverse_scale;
  }
  bool admitted() const { return admitted_; }
  double scaleForHeader(std::uint64_t header_ns) const
  {
    require(admitted_, "not_admitted");
    const auto found = scans_.find(header_ns); require(found != scans_.end(), "scan_missing");
    return found->second.inverse_scale;
  }
  const Context &context() const { return context_; }
  const std::string &method() const { return method_; }
private:
  const Scan &verify(std::uint64_t header_ns, const std::vector<std::uint8_t> &original,
                     std::uint64_t event_ns) const
  {
    require(event_ns > context_.checkpoint_ns && event_ns <= context_.decision_ns &&
            header_ns <= event_ns, "event_time");
    const auto found = scans_.find(header_ns); require(found != scans_.end(), "scan_missing");
    require(!original.empty() && sha256Bytes(original.data(), original.size()) ==
            found->second.raw_sha256, "raw_hash_mismatch");
    return found->second;
  }
  bool loaded_ = false, preflight_started_ = false, admitted_ = false, failed_ = false;
  std::size_t preflight_calls_ = 0, calls_ = 0;
  Context context_; std::string method_;
  std::map<std::uint64_t, Scan> scans_; std::set<std::uint64_t> observed_;
};
} // namespace fast_livo::scan_profile
