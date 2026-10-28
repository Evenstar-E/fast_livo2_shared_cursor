#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/point_tests.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

// The map metric uses geometry only. Loading into PointXYZ keeps the comparator
// compatible with RGB legacy maps and intensity-based bounded voxel exports
// without emitting missing-field warnings or silently depending on colour.
using Cloud = pcl::PointCloud<pcl::PointXYZ>;

struct DirectionalMetrics
{
  std::size_t count = 0;
  double mean = 0.0;
  double rmse = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double within_10cm = 0.0;
  double within_20cm = 0.0;
  double within_50cm = 0.0;
  std::size_t over_10cm_count = 0;
};

struct AlignedCloud
{
  Cloud::Ptr cloud;
  bool converged = false;
  double fitness = 0.0;
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
};

Cloud::Ptr loadCloud(const std::string &path)
{
  Cloud::Ptr cloud(new Cloud);
  if (pcl::io::loadPCDFile(path, *cloud) != 0)
    throw std::runtime_error("failed to load PCD: " + path);

  Cloud::Ptr finite(new Cloud);
  finite->reserve(cloud->size());
  for (const auto &point : cloud->points)
    if (pcl::isFinite(point)) finite->push_back(point);
  return finite;
}

double percentile(const std::vector<double> &sorted, double probability)
{
  if (sorted.empty()) return 0.0;
  const double index = probability * static_cast<double>(sorted.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(index));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
  const double alpha = index - lower;
  return sorted[lower] * (1.0 - alpha) + sorted[upper] * alpha;
}

DirectionalMetrics compareDirection(const Cloud::Ptr &reference,
                                      const Cloud::Ptr &query)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  tree.setInputCloud(reference);
  std::vector<double> distances;
  distances.reserve(query->size());
  std::vector<int> indices(1);
  std::vector<float> squared_distances(1);

  for (const auto &point : query->points)
  {
    if (tree.nearestKSearch(point, 1, indices, squared_distances) == 1)
      distances.push_back(std::sqrt(static_cast<double>(squared_distances[0])));
  }
  if (distances.empty()) throw std::runtime_error("no finite nearest-neighbour distances");

  DirectionalMetrics metrics;
  metrics.count = distances.size();
  double sum = 0.0;
  double squared_sum = 0.0;
  std::size_t n10 = 0, n20 = 0, n50 = 0;
  for (const double distance : distances)
  {
    sum += distance;
    squared_sum += distance * distance;
    if (distance <= 0.10) ++n10;
    if (distance <= 0.20) ++n20;
    if (distance <= 0.50) ++n50;
  }
  std::sort(distances.begin(), distances.end());
  metrics.mean = sum / distances.size();
  metrics.rmse = std::sqrt(squared_sum / distances.size());
  metrics.median = percentile(distances, 0.50);
  metrics.p95 = percentile(distances, 0.95);
  metrics.within_10cm = static_cast<double>(n10) / distances.size();
  metrics.within_20cm = static_cast<double>(n20) / distances.size();
  metrics.within_50cm = static_cast<double>(n50) / distances.size();
  metrics.over_10cm_count = distances.size() - n10;
  return metrics;
}

AlignedCloud alignToReference(const Cloud::Ptr &reference,
                              const Cloud::Ptr &candidate)
{
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(candidate);
  icp.setInputTarget(reference);
  icp.setMaximumIterations(60);
  icp.setMaxCorrespondenceDistance(0.75);
  icp.setTransformationEpsilon(1e-9);
  icp.setEuclideanFitnessEpsilon(1e-8);

  AlignedCloud result;
  result.cloud.reset(new Cloud);
  icp.align(*result.cloud);
  result.converged = icp.hasConverged();
  result.fitness = icp.getFitnessScore(0.75);
  result.transform = icp.getFinalTransformation();
  return result;
}

void printMetrics(const char *name, const DirectionalMetrics &metrics, bool trailing_comma)
{
  std::cout << "    \"" << name << "\": {"
            << "\"count\":" << metrics.count
            << ",\"mean_m\":" << metrics.mean
            << ",\"rmse_m\":" << metrics.rmse
            << ",\"median_m\":" << metrics.median
            << ",\"p95_m\":" << metrics.p95
            << ",\"within_10cm\":" << metrics.within_10cm
            << ",\"within_20cm\":" << metrics.within_20cm
            << ",\"within_50cm\":" << metrics.within_50cm
            << ",\"over_10cm_count\":" << metrics.over_10cm_count << "}"
            << (trailing_comma ? "," : "") << '\n';
}

void printPair(const Cloud::Ptr &clean, const Cloud::Ptr &candidate)
{
  const auto residue = compareDirection(clean, candidate);
  const auto completeness = compareDirection(candidate, clean);
  std::cout << std::fixed << std::setprecision(9) << "{\n";
  std::cout << "  \"metric_family\":\"direct_frame_bidirectional_nearest_neighbor\",\n";
  std::cout << "  \"alignment_applied\":false,\n";
  std::cout << "  \"distance_threshold_m\":0.1,\n";
  std::cout << "  \"point_counts\":{\"clean\":" << clean->size()
            << ",\"candidate\":" << candidate->size() << "},\n";
  std::cout << "  \"candidate\": {\n";
  printMetrics("candidate_to_clean", residue, true);
  printMetrics("clean_to_candidate", completeness, true);
  std::cout << "    \"candidate_to_clean_rmse_m\":" << residue.rmse << ",\n"
            << "    \"candidate_to_clean_p95_m\":" << residue.p95 << ",\n"
            << "    \"clean_to_candidate_rmse_m\":" << completeness.rmse << ",\n"
            << "    \"clean_to_candidate_p95_m\":" << completeness.p95 << ",\n"
            << "    \"extra_pollution_points\":" << residue.over_10cm_count << ",\n"
            << "    \"extra_pollution_fraction\":"
            << static_cast<double>(residue.over_10cm_count) / residue.count << ",\n"
            << "    \"coverage_within_10cm\":" << completeness.within_10cm << "\n";
  std::cout << "  }\n}\n";
}

void printCondition(const char *name, const Cloud::Ptr &clean,
                    const Cloud::Ptr &candidate, bool trailing_comma)
{
  const auto raw_residue = compareDirection(clean, candidate);
  const auto raw_completeness = compareDirection(candidate, clean);
  const auto aligned = alignToReference(clean, candidate);
  const auto aligned_residue = compareDirection(clean, aligned.cloud);
  const auto aligned_completeness = compareDirection(aligned.cloud, clean);
  std::cout << "  \"" << name << "\": {\n";
  std::cout << "    \"icp\":{\"converged\":" << (aligned.converged ? "true" : "false")
            << ",\"fitness\":" << aligned.fitness
            << ",\"translation_m\":[" << aligned.transform(0, 3) << ','
            << aligned.transform(1, 3) << ',' << aligned.transform(2, 3) << "]},\n";
  std::cout << "    \"unaligned\": {\n";
  printMetrics("candidate_to_clean", raw_residue, true);
  printMetrics("clean_to_candidate", raw_completeness, false);
  std::cout << "    },\n";
  std::cout << "    \"aligned\": {\n";
  printMetrics("candidate_to_clean", aligned_residue, true);
  printMetrics("clean_to_candidate", aligned_completeness, false);
  std::cout << "    }\n";
  std::cout << "  }" << (trailing_comma ? "," : "") << '\n';
}

} // namespace

int main(int argc, char **argv)
{
  if (argc != 3 && argc != 4)
  {
    std::cerr << "Usage: " << argv[0]
              << " CLEAN.pcd CANDIDATE.pcd\n       " << argv[0]
              << " CLEAN.pcd POLLUTED.pcd RECOVERED.pcd\n";
    return 2;
  }

  try
  {
    const Cloud::Ptr clean = loadCloud(argv[1]);
    const Cloud::Ptr polluted = loadCloud(argv[2]);
    if (argc == 3)
    {
      printPair(clean, polluted);
      return 0;
    }
    const Cloud::Ptr recovered = loadCloud(argv[3]);
    std::cout << std::fixed << std::setprecision(6) << "{\n";
    std::cout << "  \"point_counts\":{\"clean\":" << clean->size()
              << ",\"polluted\":" << polluted->size()
              << ",\"recovered\":" << recovered->size() << "},\n";
    printCondition("polluted", clean, polluted, true);
    printCondition("recovered", clean, recovered, false);
    std::cout << "}\n";
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
