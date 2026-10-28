#include "range_scale_correction_estimator.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using fast_livo::correction_estimation::Config;
using fast_livo::correction_estimation::HypothesisEvidence;
using fast_livo::correction_estimation::OnsetInterval;
using fast_livo::correction_estimation::RangeScaleCorrectionEstimator;

double number(const char *text, const char *label)
{
  char *end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value))
    throw std::invalid_argument(std::string("invalid ") + label);
  return value;
}

std::size_t count(const char *text, const char *label)
{
  const double value = number(text, label);
  if (value < 0.0 || std::floor(value) != value)
    throw std::invalid_argument(std::string("invalid ") + label);
  return static_cast<std::size_t>(value);
}

HypothesisEvidence hypothesis(const std::string &text)
{
  std::vector<std::string> fields;
  std::stringstream stream(text);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  if (fields.size() != 6)
    throw std::invalid_argument(
        "hypothesis must be scale,residual,match,innovation,samples,healthy");
  HypothesisEvidence value;
  value.range_scale = number(fields[0].c_str(), "hypothesis scale");
  value.residual_median = number(fields[1].c_str(), "hypothesis residual");
  value.match_ratio_median = number(fields[2].c_str(), "hypothesis match");
  value.innovation_median = number(fields[3].c_str(), "hypothesis innovation");
  value.sample_count = count(fields[4].c_str(), "hypothesis sample count");
  if (fields[5] != "0" && fields[5] != "1")
    throw std::invalid_argument("hypothesis healthy flag must be 0 or 1");
  value.healthy_tail_pass = fields[5] == "1";
  return value;
}

std::string json(const fast_livo::correction_estimation::Estimate &result)
{
  const auto value = [](const double number)
  {
    if (!std::isfinite(number)) return std::string("null");
    std::ostringstream stream;
    stream << std::setprecision(17) << number;
    return stream.str();
  };
  std::ostringstream output;
  output << "{\n"
      << "  \"schema\": 1,\n"
      << "  \"accepted\": " << (result.accepted ? "true" : "false") << ",\n"
      << "  \"decision\": \""
      << (result.accepted ? "candidate" : "identity") << "\",\n"
      << "  \"reason\": \"" << result.reason << "\",\n"
      << "  \"diagnostic_winner_range_scale\": "
      << value(result.range_scale) << ",\n"
      << "  \"selector_action_range_scale\": "
      << value(result.accepted ? result.range_scale : 1.0) << ",\n"
      << "  \"range_scale\": " << value(result.range_scale) << ",\n"
      << "  \"inverse_range_scale\": " << value(result.inverse_range_scale) << ",\n"
      << "  \"range_scale_lower\": " << value(result.range_scale_lower) << ",\n"
      << "  \"range_scale_upper\": " << value(result.range_scale_upper) << ",\n"
      << "  \"confidence\": " << value(result.confidence) << ",\n"
      << "  \"winner_score\": " << value(result.winner_score) << ",\n"
      << "  \"identity_score\": " << value(result.identity_score) << ",\n"
      << "  \"relative_improvement\": "
      << value(result.relative_improvement) << ",\n"
      << "  \"runner_up_margin\": " << value(result.runner_up_margin) << ",\n"
      << "  \"eligible_hypotheses\": "
      << result.eligible_hypotheses << ",\n"
      << "  \"winner_samples\": " << result.winner_samples << "\n"
      << "}\n";
  return output.str();
}
} // namespace

int main(int argc, char **argv)
{
  try
  {
    Config config;
    config.enabled = true;
    OnsetInterval onset;
    std::vector<HypothesisEvidence> hypotheses;
    std::string output_path;
    for (int index = 1; index < argc; ++index)
    {
      const std::string key(argv[index]);
      if (index + 1 >= argc)
        throw std::invalid_argument("missing value for " + key);
      const char *value = argv[++index];
      if (key == "--onset-lower") onset.lower_bound_sec = number(value, "onset lower");
      else if (key == "--onset-point") onset.point_estimate_sec = number(value, "onset point");
      else if (key == "--onset-upper") onset.upper_bound_sec = number(value, "onset upper");
      else if (key == "--onset-confidence") onset.confidence = number(value, "onset confidence");
      else if (key == "--detection") onset.detection_sec = number(value, "detection");
      else if (key == "--minimum-samples") config.minimum_samples = count(value, "minimum samples");
      else if (key == "--minimum-hypotheses") config.minimum_hypotheses = count(value, "minimum hypotheses");
      else if (key == "--minimum-improvement") config.minimum_relative_improvement = number(value, "minimum improvement");
      else if (key == "--minimum-margin") config.minimum_runner_up_margin = number(value, "minimum margin");
      else if (key == "--minimum-confidence") config.minimum_combined_confidence = number(value, "minimum confidence");
      else if (key == "--hypothesis") hypotheses.push_back(hypothesis(value));
      else if (key == "--output") output_path = value;
      else throw std::invalid_argument("unknown option: " + key);
    }
    if (output_path.empty())
      throw std::invalid_argument("--output is required");
    const RangeScaleCorrectionEstimator estimator(config);
    const auto result = estimator.estimate(onset, hypotheses);
    const std::string document = json(result);
    std::ofstream output(output_path, std::ios::out | std::ios::trunc);
    if (!output.is_open()) throw std::runtime_error("cannot open output file");
    output << document;
    output.close();
    std::cout << document;
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "range_scale_correction_selector: " << error.what() << '\n';
    return 2;
  }
}
