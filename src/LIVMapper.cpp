/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include "checkpoint_binary_codec.h"
#include "replay_ingest_carry_guard.h"
#include "replay_commit_boundary.h"
#include "source_revision_hash.h"
#include "private_map_receipt.h"
#include "typed_event_map_snapshot.h"
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <unordered_map>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace
{
using fast_livo::checkpoint_io::Reader;
using fast_livo::checkpoint_io::Writer;

constexpr std::uint64_t kMapperCheckpointMagic = 0x31504b4350414d46ULL; // FMAPCKP1
constexpr std::uint32_t kMapperCheckpointSchemaVersion = 2;
constexpr std::uint64_t kRecoveryHandoffMagic = 0x31464f444e414846ULL; // FHANDOF1
constexpr std::uint32_t kRecoveryHandoffSchemaVersion = 1;

bool sourceRevisionReceiptEnabled()
{
  const char *value = std::getenv("SHADOW_SOURCE_REVISION_RECEIPT_ENABLED");
  return value != nullptr && std::strcmp(value, "true") == 0;
}

bool sourceRevisionCutEnabled()
{
  const char *value = std::getenv("SHADOW_SOURCE_REVISION_CUT_ENABLED");
  return value != nullptr && std::strcmp(value, "true") == 0;
}

bool sourceRevisionBarrierEnabled()
{
  const char *value = std::getenv("SHADOW_SOURCE_REVISION_BARRIER_ENABLED");
  return value != nullptr && std::strcmp(value, "true") == 0;
}

bool sourceRevisionPayloadHashEnabled()
{
  const char *value = std::getenv("SHADOW_SOURCE_REVISION_PAYLOAD_HASH_ENABLED");
  return value != nullptr && std::strcmp(value, "true") == 0;
}

std::uint64_t p2SensorStampNs(const LidarMeasureGroup &group)
{
  // These are sensor timestamps already present in the completed sync group.
  // Their maximum is only an input-availability upper bound at consumption;
  // it is NOT the ROS callback-arrival time of any constituent message.
  double latest = 0.0;
  const auto consider = [&latest](double seconds) {
    if (std::isfinite(seconds) && seconds > latest) latest = seconds;
  };
  consider(group.lidar_frame_beg_time);
  consider(group.lidar_frame_end_time);
  consider(group.last_lio_update_time);
  for (const auto &measure : group.measures)
  {
    consider(measure.lio_time);
    consider(measure.vio_time);
    for (const auto &imu : measure.imu)
      if (imu) consider(imu->header.stamp.toSec());
  }
  for (const auto &raw : group.raw_lidar_frames)
  {
    consider(raw.sensor_header_time);
    consider(raw.effective_header_time);
  }
  if (latest <= 0.0 || latest >= 18446744073.0) return 0;
  return static_cast<std::uint64_t>(std::llround(latest * 1e9));
}

std::string p2FiniteSeconds(double value)
{
  if (!std::isfinite(value)) return "";
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

bool syncFilePath(const std::string &path)
{
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  const bool ok = ::fsync(fd) == 0;
  ::close(fd);
  return ok;
}

bool syncDirectoryPath(const std::string &path)
{
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
  if (fd < 0) return false;
  const bool ok = ::fsync(fd) == 0;
  ::close(fd);
  return ok;
}

void writeDetectorSample(
    Writer &writer,
    const fast_livo::harm_detection::OnlineHarmDetector::Sample &sample)
{
  writer.pod(sample.time_sec);
  writer.pod(sample.residual);
  writer.pod(sample.match_ratio);
  writer.pod(sample.innovation);
  writer.pod(static_cast<std::uint8_t>(sample.weak_evidence));
}

fast_livo::harm_detection::OnlineHarmDetector::Sample readDetectorSample(
    Reader &reader)
{
  fast_livo::harm_detection::OnlineHarmDetector::Sample sample;
  sample.time_sec = reader.pod<double>();
  sample.residual = reader.pod<double>();
  sample.match_ratio = reader.pod<double>();
  sample.innovation = reader.pod<double>();
  sample.weak_evidence = reader.pod<std::uint8_t>() != 0;
  return sample;
}

void writeDetectorSamples(
    Writer &writer,
    const std::deque<fast_livo::harm_detection::OnlineHarmDetector::Sample> &samples)
{
  writer.pod(static_cast<std::uint64_t>(samples.size()));
  for (const auto &sample : samples) writeDetectorSample(writer, sample);
}

std::deque<fast_livo::harm_detection::OnlineHarmDetector::Sample>
readDetectorSamples(Reader &reader, const char *label)
{
  const std::uint64_t count = fast_livo::checkpoint_io::checkedCount(
      reader, 100000000, label);
  std::deque<fast_livo::harm_detection::OnlineHarmDetector::Sample> samples;
  for (std::uint64_t index = 0; index < count; ++index)
    samples.push_back(readDetectorSample(reader));
  return samples;
}

void writeDetectorSnapshot(
    Writer &writer,
    const fast_livo::harm_detection::OnlineHarmDetector::Snapshot &snapshot)
{
  writeDetectorSamples(writer, snapshot.baseline);
  writeDetectorSamples(writer, snapshot.long_history);
  writeDetectorSamples(writer, snapshot.evidence_history);
  writer.pod(static_cast<std::uint64_t>(snapshot.sample_intervals_sec.size()));
  for (const double value : snapshot.sample_intervals_sec) writer.pod(value);
  writer.pod(static_cast<std::uint64_t>(snapshot.consecutive_anomalies));
  writer.pod(snapshot.candidate_onset_sec);
  writer.pod(snapshot.last_sample_time_sec);
  writer.pod(static_cast<std::uint8_t>(snapshot.triggered));
  writer.pod(static_cast<std::uint8_t>(snapshot.suppressed_rearm_pending));
  writer.pod(static_cast<std::uint64_t>(snapshot.episode_id));
  writer.pod(static_cast<std::uint64_t>(snapshot.episode_proposal_count));
  writer.pod(snapshot.last_episode_detection_sec);
  writer.pod(snapshot.last_episode_lower_bound_sec);
  writer.pod(snapshot.last_episode_upper_bound_sec);
}

fast_livo::harm_detection::OnlineHarmDetector::Snapshot readDetectorSnapshot(
    Reader &reader)
{
  fast_livo::harm_detection::OnlineHarmDetector::Snapshot snapshot;
  snapshot.baseline = readDetectorSamples(reader, "detector baseline sample");
  snapshot.long_history = readDetectorSamples(reader, "detector long-history sample");
  snapshot.evidence_history = readDetectorSamples(reader, "detector evidence sample");
  const std::uint64_t interval_count = fast_livo::checkpoint_io::checkedCount(
      reader, 100000000, "detector interval");
  for (std::uint64_t index = 0; index < interval_count; ++index)
    snapshot.sample_intervals_sec.push_back(reader.pod<double>());
  snapshot.consecutive_anomalies =
      static_cast<std::size_t>(reader.pod<std::uint64_t>());
  snapshot.candidate_onset_sec = reader.pod<double>();
  snapshot.last_sample_time_sec = reader.pod<double>();
  snapshot.triggered = reader.pod<std::uint8_t>() != 0;
  snapshot.suppressed_rearm_pending = reader.pod<std::uint8_t>() != 0;
  snapshot.episode_id = static_cast<std::size_t>(reader.pod<std::uint64_t>());
  snapshot.episode_proposal_count =
      static_cast<std::size_t>(reader.pod<std::uint64_t>());
  snapshot.last_episode_detection_sec = reader.pod<double>();
  snapshot.last_episode_lower_bound_sec = reader.pod<double>();
  snapshot.last_episode_upper_bound_sec = reader.pod<double>();
  return snapshot;
}

void writeBlob(Writer &writer, const std::vector<std::uint8_t> &blob)
{
  writer.pod(static_cast<std::uint64_t>(blob.size()));
  writer.raw(blob.data(), blob.size());
}

struct ByteView
{
  const std::uint8_t *data = nullptr;
  std::size_t size = 0;
};

ByteView readBlobView(Reader &reader, const char *label)
{
  const std::uint64_t size = fast_livo::checkpoint_io::checkedCount(
      reader, static_cast<std::uint64_t>(reader.remaining()), label);
  Reader view = reader.subreader(static_cast<std::size_t>(size));
  return {view.data(), view.size()};
}

class ReadOnlyFileMapping
{
public:
  explicit ReadOnlyFileMapping(const std::string &path)
  {
    descriptor_ = ::open(path.c_str(), O_RDONLY);
    if (descriptor_ < 0)
      throw std::runtime_error("cannot open recovery handoff bundle: " +
                               std::string(std::strerror(errno)));
    struct stat status;
    if (::fstat(descriptor_, &status) != 0)
    {
      const int saved_errno = errno;
      ::close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("cannot stat recovery handoff bundle: " +
                               std::string(std::strerror(saved_errno)));
    }
    if (status.st_size <= 0)
    {
      ::close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("empty recovery handoff bundle");
    }
    size_ = static_cast<std::size_t>(status.st_size);
    mapping_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
    if (mapping_ == MAP_FAILED)
    {
      mapping_ = nullptr;
      const int saved_errno = errno;
      ::close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("cannot map recovery handoff bundle: " +
                               std::string(std::strerror(saved_errno)));
    }
  }

  ~ReadOnlyFileMapping()
  {
    if (mapping_ != nullptr) ::munmap(mapping_, size_);
    if (descriptor_ >= 0) ::close(descriptor_);
  }

  ReadOnlyFileMapping(const ReadOnlyFileMapping &) = delete;
  ReadOnlyFileMapping &operator=(const ReadOnlyFileMapping &) = delete;

  const std::uint8_t *data() const
  {
    return static_cast<const std::uint8_t *>(mapping_);
  }
  std::size_t size() const { return size_; }

private:
  int descriptor_ = -1;
  void *mapping_ = nullptr;
  std::size_t size_ = 0;
};

void writeState(Writer &writer, const StatesGroup &state)
{
  writer.eigen(state.rot_end);
  writer.eigen(state.pos_end);
  writer.eigen(state.vel_end);
  writer.pod(state.inv_expo_time);
  writer.eigen(state.bias_g);
  writer.eigen(state.bias_a);
  writer.eigen(state.gravity);
  writer.eigen(state.cov);
}

StatesGroup readState(Reader &reader)
{
  StatesGroup state;
  reader.eigen(state.rot_end);
  reader.eigen(state.pos_end);
  reader.eigen(state.vel_end);
  state.inv_expo_time = reader.pod<double>();
  reader.eigen(state.bias_g);
  reader.eigen(state.bias_a);
  reader.eigen(state.gravity);
  reader.eigen(state.cov);
  return state;
}

void writeMapperPointWithVar(Writer &writer, const pointWithVar &point)
{
  writer.eigen(point.point_b);
  writer.eigen(point.point_i);
  writer.eigen(point.point_w);
  writer.eigen(point.var_nostate);
  writer.eigen(point.body_var);
  writer.eigen(point.var);
  writer.eigen(point.point_crossmat);
  writer.eigen(point.normal);
}

pointWithVar readMapperPointWithVar(Reader &reader)
{
  pointWithVar point;
  reader.eigen(point.point_b);
  reader.eigen(point.point_i);
  reader.eigen(point.point_w);
  reader.eigen(point.var_nostate);
  reader.eigen(point.body_var);
  reader.eigen(point.var);
  reader.eigen(point.point_crossmat);
  reader.eigen(point.normal);
  return point;
}

void writeMapperPoint(Writer &writer, const PointType &point)
{
  writer.pod(point.x);
  writer.pod(point.y);
  writer.pod(point.z);
  writer.pod(point.intensity);
  writer.pod(point.normal_x);
  writer.pod(point.normal_y);
  writer.pod(point.normal_z);
  writer.pod(point.curvature);
}

PointType readMapperPoint(Reader &reader)
{
  PointType point{};
  point.x = reader.pod<float>();
  point.y = reader.pod<float>();
  point.z = reader.pod<float>();
  point.intensity = reader.pod<float>();
  point.normal_x = reader.pod<float>();
  point.normal_y = reader.pod<float>();
  point.normal_z = reader.pod<float>();
  point.curvature = reader.pod<float>();
  return point;
}

void writeMapperCloud(Writer &writer, const PointCloudXYZI &cloud)
{
  writer.pod(cloud.header.seq);
  writer.pod(cloud.header.stamp);
  writer.string(cloud.header.frame_id);
  writer.pod(cloud.width);
  writer.pod(cloud.height);
  writer.pod(static_cast<std::uint8_t>(cloud.is_dense));
  writer.eigen(cloud.sensor_origin_);
  writer.eigen(cloud.sensor_orientation_.coeffs());
  writer.pod(static_cast<std::uint64_t>(cloud.points.size()));
  for (const PointType &point : cloud.points) writeMapperPoint(writer, point);
}

PointCloudXYZI readMapperCloud(Reader &reader, bool require_rectangular_shape = true)
{
  PointCloudXYZI cloud;
  cloud.header.seq = reader.pod<std::uint32_t>();
  cloud.header.stamp = reader.pod<std::uint64_t>();
  cloud.header.frame_id = reader.string();
  cloud.width = reader.pod<std::uint32_t>();
  cloud.height = reader.pod<std::uint32_t>();
  cloud.is_dense = reader.pod<std::uint8_t>() != 0;
  reader.eigen(cloud.sensor_origin_);
  Eigen::Vector4f orientation;
  reader.eigen(orientation);
  cloud.sensor_orientation_.coeffs() = orientation;
  const std::uint64_t count = fast_livo::checkpoint_io::checkedCount(
      reader, 1000000000, "mapper cloud point");
  cloud.points.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index)
    cloud.points.push_back(readMapperPoint(reader));
  if (require_rectangular_shape && cloud.points.size() != static_cast<std::size_t>(cloud.width) * cloud.height)
    throw std::runtime_error("mapper checkpoint cloud shape mismatch");
  return cloud;
}

double robustMedian(std::deque<double> values)
{
  if (values.empty()) return 0.0;
  std::vector<double> sorted(values.begin(), values.end());
  const std::size_t middle = sorted.size() / 2;
  std::nth_element(sorted.begin(), sorted.begin() + middle, sorted.end());
  const double upper = sorted[middle];
  if (sorted.size() % 2 != 0) return upper;
  std::nth_element(sorted.begin(), sorted.begin() + middle - 1, sorted.begin() + middle);
  return 0.5 * (sorted[middle - 1] + upper);
}

double robustMad(const std::deque<double> &values, const double center)
{
  std::deque<double> deviations;
  for (const double value : values) deviations.push_back(std::abs(value - center));
  return 1.4826 * robustMedian(deviations);
}

std::string joinPath(const std::string &directory, const std::string &filename)
{
  if (directory.empty()) return filename;
  const char last = directory.back();
  return directory + ((last == '/' || last == '\\') ? "" : "/") + filename;
}
} // namespace

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),
      extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(nh);
  initializeTransactionDiagnostics();
  map_transaction_state.reset(new fast_livo::transactional_mapping::StateMachine(map_transaction_config));
  harm_detector.reset(new fast_livo::harm_detection::OnlineHarmDetector(harm_detector_config));
  recovery_proposal_arbiter.reset(
      new fast_livo::recovery_scheduling::RecoveryProposalArbiter(
          recovery_proposal_arbiter_config));
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  voxelmap_manager->selective_gate_config_ = selective_lidar_gate_config;
  voxelmap_manager->robust_weight_config_ = robust_lidar_weight_config;
  vio_manager.reset(new VIOManager());
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents();
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  std::string fusion_audit_path;
  nh.param<std::string>("diagnostic/fusion_audit_path", fusion_audit_path, "");
  fusion_audit.open(fusion_audit_path);
  std::string p2_runtime_receipt_path;
  nh.param<std::string>("diagnostic/p2_runtime_receipt_path",
                        p2_runtime_receipt_path, "");
  if (!p2_runtime_receipt_path.empty())
  {
    // Receipt event IDs have meaning only alongside the actual fusion rows.
    if (fusion_audit_path.empty() || !std::ifstream(fusion_audit_path).good())
      throw std::runtime_error(
          "p2 receipt requires an open diagnostic/fusion_audit_path");
    p2_runtime_receipt.open(p2_runtime_receipt_path);
  }
  std::string p2_live_metric_receipt_path;
  nh.param<std::string>("diagnostic/p2_live_metric_receipt_path",
                        p2_live_metric_receipt_path, "");
  if (!p2_live_metric_receipt_path.empty())
  {
    if (!p2_runtime_receipt.enabled())
      throw std::runtime_error("p2 live metric requires p2 runtime receipt and fusion audit");
    p2_live_metric_receipt.open(p2_live_metric_receipt_path);
  }
  std::string input_event_profile_path;
  nh.param<std::string>("diagnostic/input_event_profile_path",
                        input_event_profile_path, "");
  input_event_profile.open(input_event_profile_path);
  nh.param<bool>("diagnostic/forbid_recovery", diagnostic_forbid_recovery, false);
  nh.param<bool>("diagnostic/vio_patch_boundary_check_only", vio_patch_boundary_check_only, false);
  nh.param<bool>("vio/patch_boundary_guard", vio_patch_boundary_guard, true);
  nh.param<std::string>("diagnostic/nominal_probe_directory", nominal_probe_directory, "");
  nh.param<double>("diagnostic/nominal_probe_start", nominal_probe_start, 10.0);
  nh.param<double>("diagnostic/nominal_probe_end", nominal_probe_end, 12.0);
  nh.param<std::string>("diagnostic/causal_event_archive_directory", causal_event_archive_directory, "");
  nh.param<std::string>("diagnostic/causal_reference_event_directory", causal_reference_event_directory, "");
  nh.param<std::string>("diagnostic/causal_cloud_audit_directory", causal_cloud_audit_directory, "");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");

  nh.param<bool>("vio/normal_en", normal_en, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>("vio/max_iterations", max_iterations, 5);
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<int>("vio/grid_size", grid_size, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  nh.param<int>("vio/patch_size", patch_size, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<bool>("preprocess/hilti_en", hilti_en, false);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<int>("pcd_save/type", pcd_save_type, 0);
  nh.param<bool>("image_save/img_save_en", img_save_en, false);
  nh.param<int>("image_save/interval", img_save_interval, 1);

  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<std::string>("pcd_save/output_directory", pcd_output_directory,
                        std::string(ROOT_DIR) + "Log/pcd");
  nh.param<bool>("pcd_save/bounded_voxel_export_en",
                 bounded_voxel_export_en, false);
  int bounded_export_max_points =
      static_cast<int>(bounded_voxel_export_max_points);
  int bounded_export_samples_per_plane =
      static_cast<int>(bounded_voxel_export_samples_per_plane);
  nh.param<int>("pcd_save/bounded_voxel_export_max_points",
                bounded_export_max_points, bounded_export_max_points);
  nh.param<int>("pcd_save/bounded_voxel_export_samples_per_plane",
                bounded_export_samples_per_plane,
                bounded_export_samples_per_plane);
  bounded_voxel_export_max_points = static_cast<std::size_t>(
      std::max(1, bounded_export_max_points));
  bounded_voxel_export_samples_per_plane = static_cast<std::size_t>(
      std::max(1, std::min(9, bounded_export_samples_per_plane)));
  nh.param<bool>("shadow_online_map_publish/enabled",
                 shadow_online_map_publish_enabled, false);
  nh.param<double>("shadow_online_map_publish/period_sec",
                   shadow_online_map_publish_period_sec, 10.0);
  int online_map_max_points = static_cast<int>(
      shadow_online_map_publish_max_points);
  nh.param<int>("shadow_online_map_publish/max_points",
                online_map_max_points, online_map_max_points);
  shadow_online_map_publish_max_points = static_cast<std::size_t>(
      std::max(1, online_map_max_points));
  if (shadow_online_map_publish_enabled &&
      shadow_online_map_publish_period_sec <= 0.0)
    throw std::invalid_argument("online map publication period must be positive");
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);

  nh.param<bool>("transactional_mapping/enabled", map_transaction_config.enabled, false);
  nh.param<double>("transactional_mapping/checkpoint_start_sec", map_transaction_config.checkpoint_start_sec, -1.0);
  nh.param<double>("transactional_mapping/oracle_start_sec", map_transaction_config.quarantine_start_sec, -1.0);
  nh.param<double>("transactional_mapping/oracle_end_sec", map_transaction_config.quarantine_end_sec, -1.0);
  nh.param<double>("transactional_mapping/detection_delay_sec", map_transaction_config.detection_delay_sec, 0.0);
  nh.param<double>("transactional_mapping/recovery_tail_sec", map_transaction_config.recovery_tail_sec, 1.0);
  int max_buffered_events = static_cast<int>(map_transaction_config.max_buffered_events);
  nh.param<int>("transactional_mapping/max_buffered_events", max_buffered_events, max_buffered_events);
  map_transaction_config.max_buffered_events = static_cast<std::size_t>(std::max(1, max_buffered_events));
  nh.param<bool>("transactional_mapping/live_lidar_provisional_writes", live_lidar_provisional_writes, true);
  nh.param<bool>("transactional_mapping/delta_voxel_journal_enabled",
                 delta_voxel_journal_enabled, false);
  nh.param<bool>("transactional_mapping/freeze_vio_provisional", freeze_vio_provisional, true);
  nh.param<bool>("transactional_mapping/replay_lidar_updates_in_quarantine", replay_lidar_updates_in_quarantine, false);
  nh.param<bool>("transactional_mapping/replay_visual_updates_in_quarantine", replay_visual_updates_in_quarantine, false);
  nh.param<bool>("transactional_mapping/replay_map_writes_in_quarantine", replay_map_writes_in_quarantine, false);
  nh.param<bool>("transactional_mapping/replay_map_writes_during_validation_tail",
                 replay_map_writes_during_validation_tail, true);
  nh.param<bool>("transactional_mapping/selective_lidar_gate_during_validation_tail",
                 selective_lidar_gate_during_validation_tail, false);
  nh.param<bool>("transactional_mapping/selective_map_write_gate_during_validation_tail",
                 selective_map_write_gate_during_validation_tail, false);
  nh.param<bool>("transactional_mapping/replay_reanchor_pose_to_incumbent_at_commit",
                 replay_reanchor_pose_to_incumbent_at_commit, false);
  nh.param<bool>("transactional_mapping/output_boundary_alignment_enabled",
                 replay_output_boundary_alignment_enabled, false);
  nh.param<bool>("transactional_mapping/replay_pcd_inliers_in_quarantine",
                 replay_pcd_inliers_in_quarantine, false);
  nh.param<double>("transactional_mapping/replay_fault_lidar_weight", replay_fault_lidar_weight, 1.0);
  if (replay_fault_lidar_weight <= 0.0 || replay_fault_lidar_weight > 1.0)
    throw std::invalid_argument("replay fault LiDAR weight must be in (0, 1]");
  nh.param<bool>("transactional_mapping/oracle_range_scale_correction_enabled",
                 replay_range_scale_oracle_config.enabled, false);
  nh.param<double>("transactional_mapping/oracle_range_scale_start_sec",
                   replay_range_scale_oracle_config.start_sec, 0.0);
  nh.param<double>("transactional_mapping/oracle_range_scale_end_sec",
                   replay_range_scale_oracle_config.end_sec, 0.0);
  nh.param<double>("transactional_mapping/oracle_range_scale_start",
                   replay_range_scale_oracle_config.scale_start, 1.0);
  nh.param<double>("transactional_mapping/oracle_range_scale_end",
                   replay_range_scale_oracle_config.scale_end, 1.0);
  replay_range_scale_oracle_config.validate();
  nh.param<bool>("diagnostic/raw_scan_time_oracle_enabled",
                 private_raw_scan_oracle_config.enabled, false);
  nh.param<double>("diagnostic/raw_scan_time_oracle_start_sec",
                   private_raw_scan_oracle_config.start_sec, 0.0);
  nh.param<double>("diagnostic/raw_scan_time_oracle_end_sec",
                   private_raw_scan_oracle_config.end_sec, 0.0);
  nh.param<double>("diagnostic/raw_scan_time_oracle_scale_start",
                   private_raw_scan_oracle_config.scale_start, 1.0);
  nh.param<double>("diagnostic/raw_scan_time_oracle_scale_end",
                   private_raw_scan_oracle_config.scale_end, 1.0);
  private_raw_scan_oracle_config.validate();
  nh.param<bool>("transactional_mapping/raw_lidar_reprocess_enabled",
                 raw_lidar_reprocess_enabled, false);
  nh.param<bool>(
      "transactional_mapping/estimated_range_scale_candidate_enabled",
      estimated_range_scale_candidate_enabled, false);
  nh.param<double>(
      "transactional_mapping/estimated_range_scale_candidate_terminal_scale",
      estimated_range_scale_candidate_terminal_scale, 1.0);
  nh.param<double>(
      "transactional_mapping/estimated_range_scale_minimum_onset_confidence",
      estimated_range_scale_minimum_onset_confidence, 0.60);
  nh.param<double>(
      "transactional_mapping/estimated_range_scale_post_detection_horizon_sec",
      estimated_range_scale_post_detection_horizon_sec, 8.0);
  nh.param<double>(
      "transactional_mapping/estimated_range_scale_candidate_ramp_end_offset_sec",
      estimated_range_scale_candidate_ramp_end_offset_sec, 0.0);
  nh.param<std::string>(
      "transactional_mapping/estimated_range_scale_candidate_ramp_start_anchor",
      estimated_range_scale_candidate_ramp_start_anchor,
      std::string("lower"));
  if (estimated_range_scale_candidate_enabled &&
      replay_range_scale_oracle_config.enabled)
    throw std::invalid_argument(
        "estimated and oracle range-scale correction are mutually exclusive");
  if (estimated_range_scale_candidate_enabled && !raw_lidar_reprocess_enabled)
    throw std::invalid_argument(
        "estimated range-scale candidate requires raw LiDAR reprocessing");
  if (!std::isfinite(estimated_range_scale_candidate_terminal_scale) ||
      estimated_range_scale_candidate_terminal_scale < 0.5 ||
      estimated_range_scale_candidate_terminal_scale > 2.0)
    throw std::invalid_argument(
        "estimated range-scale candidate terminal scale must be in [0.5, 2.0]");
  if (!std::isfinite(estimated_range_scale_minimum_onset_confidence) ||
      estimated_range_scale_minimum_onset_confidence < 0.0 ||
      estimated_range_scale_minimum_onset_confidence > 1.0)
    throw std::invalid_argument(
        "estimated range-scale onset confidence must be in [0, 1]");
  if (!std::isfinite(estimated_range_scale_post_detection_horizon_sec) ||
      estimated_range_scale_post_detection_horizon_sec <= 0.0)
    throw std::invalid_argument(
        "estimated range-scale correction horizon must be positive");
  if (!std::isfinite(estimated_range_scale_candidate_ramp_end_offset_sec) ||
      estimated_range_scale_candidate_ramp_end_offset_sec < 0.0 ||
      estimated_range_scale_candidate_ramp_end_offset_sec >
          estimated_range_scale_post_detection_horizon_sec)
    throw std::invalid_argument(
        "estimated range-scale ramp-end offset must be within the correction horizon");
  if (estimated_range_scale_candidate_ramp_start_anchor != "lower" &&
      estimated_range_scale_candidate_ramp_start_anchor != "point" &&
      estimated_range_scale_candidate_ramp_start_anchor != "upper")
    throw std::invalid_argument(
        "estimated range-scale ramp-start anchor must be lower, point, or upper");
  nh.param<std::string>("transactional_mapping/raw_lidar_reprocess_audit_path",
                        raw_lidar_reprocess_audit_path, std::string());
  if (!raw_lidar_reprocess_audit_path.empty())
  {
    raw_lidar_reprocess_audit.open(raw_lidar_reprocess_audit_path,
                                   std::ios::out | std::ios::trunc);
    if (!raw_lidar_reprocess_audit.is_open())
      throw std::runtime_error("cannot open raw LiDAR reprocess audit: " +
                               raw_lidar_reprocess_audit_path);
    raw_lidar_reprocess_audit
        << "event_relative_time,frame_count,raw_bytes,raw_hash,corrected_raw_hash,"
           "corrected_points,processed_points,processed_hash,status\n";
    raw_lidar_reprocess_audit << std::setprecision(17);
  }
  nh.param<std::string>("transactional_mapping/diagnostics_path", transaction_diagnostics_path, std::string());
  nh.param<bool>("transactional_mapping/disk_backed_replay_payload_enabled",
                 disk_backed_replay_payload_enabled, false);
  nh.param<bool>("transactional_mapping/async_replay_payload_spill_enabled",
                 async_replay_payload_spill_enabled, false);
  int async_replay_payload_queue_mib = 64;
  nh.param<int>("transactional_mapping/async_replay_payload_queue_mib",
                async_replay_payload_queue_mib, 64);
  if (async_replay_payload_spill_enabled &&
      (!disk_backed_replay_payload_enabled ||
       async_replay_payload_queue_mib < 1 ||
       async_replay_payload_queue_mib > 1024))
    throw std::invalid_argument(
        "async replay spill requires disk backing and queue size 1..1024 MiB");
  async_replay_payload_queue_bytes =
      static_cast<std::uint64_t>(async_replay_payload_queue_mib) * 1048576ULL;
  nh.param<std::string>("transactional_mapping/replay_payload_journal_path",
                        replay_payload_journal_path, std::string());
  if (disk_backed_replay_payload_enabled && replay_payload_journal_path.empty())
    throw std::invalid_argument(
        "disk-backed replay payload requires replay_payload_journal_path");
  nh.param<bool>(
      "transactional_mapping/rejected_candidate_map_export_enabled",
      rejected_candidate_map_export_enabled, false);
  nh.param<bool>("diagnostic/private_candidate_map_receipt_enabled",
                 private_candidate_map_receipt_enabled, false);
  nh.param<std::string>("diagnostic/private_candidate_map_receipt_directory",
                        private_candidate_map_receipt_directory, std::string());
  if (private_candidate_map_receipt_enabled &&
      private_candidate_map_receipt_directory.empty())
    throw std::invalid_argument("private candidate receipt requires an output directory");
  nh.param<std::string>(
      "transactional_mapping/rejected_candidate_map_export_directory",
      rejected_candidate_map_export_directory, std::string());
  int rejected_candidate_map_export_max_points = static_cast<int>(
      this->rejected_candidate_map_export_max_points);
  int rejected_candidate_map_export_samples_per_plane = static_cast<int>(
      this->rejected_candidate_map_export_samples_per_plane);
  nh.param<int>(
      "transactional_mapping/rejected_candidate_map_export_max_points",
      rejected_candidate_map_export_max_points,
      rejected_candidate_map_export_max_points);
  nh.param<int>(
      "transactional_mapping/rejected_candidate_map_export_samples_per_plane",
      rejected_candidate_map_export_samples_per_plane,
      rejected_candidate_map_export_samples_per_plane);
  this->rejected_candidate_map_export_max_points = static_cast<std::size_t>(
      std::max(1, rejected_candidate_map_export_max_points));
  this->rejected_candidate_map_export_samples_per_plane =
      static_cast<std::size_t>(std::max(
          1, std::min(9, rejected_candidate_map_export_samples_per_plane)));
  if (rejected_candidate_map_export_enabled &&
      rejected_candidate_map_export_directory.empty())
    throw std::invalid_argument(
        "rejected candidate map export requires an output directory");
  nh.param<bool>("transactional_mapping/private_fork_bind_journal",
                 private_fork_bind_journal, false);
  nh.param<bool>("transactional_mapping/private_fork_require_readonly_journal",
                 private_fork_require_readonly_journal, false);
  nh.param<std::string>("transactional_mapping/private_fork_sealed_journal_path",
                        private_fork_sealed_journal_path, std::string());
  if (private_fork_bind_journal &&
      (!disk_backed_replay_payload_enabled || replay_payload_journal_path.empty()))
    throw std::invalid_argument(
        "private fork journal binding requires a disk-backed replay journal");
  if (private_fork_require_readonly_journal &&
      private_fork_sealed_journal_path.empty())
    throw std::invalid_argument(
        "private fork import requires the sealed read-only journal path");
  nh.param<double>("diagnostic/private_fork_sealed_time_origin_sec",
                   private_fork_sealed_time_origin_sec,
                   std::numeric_limits<double>::quiet_NaN());
  if (private_fork_require_readonly_journal &&
      !std::isfinite(private_fork_sealed_time_origin_sec))
    throw std::invalid_argument(
        "private fork import requires the sealed lidar epoch");
  nh.param<double>("diagnostic/private_fork_completed_event_time_sec",
                   private_fork_completed_event_time_sec,
                   std::numeric_limits<double>::quiet_NaN());
  nh.param<int>("diagnostic/private_fork_next_event_type",
                private_fork_next_event_type, 0);
  nh.param<double>("diagnostic/private_fork_next_event_absolute_time_sec",
                   private_fork_next_event_absolute_time_sec,
                   std::numeric_limits<double>::quiet_NaN());
  nh.param<std::string>("diagnostic/private_fork_next_event_cur_hash",
                        private_fork_next_event_cur_hash, std::string());
  nh.param<std::string>("diagnostic/private_fork_next_event_image_hash",
                        private_fork_next_event_image_hash, std::string());
  nh.param<std::string>("diagnostic/private_fork_next_event_imu_hash",
                        private_fork_next_event_imu_hash, std::string());
  nh.param<int>("diagnostic/private_fork_next_event_cur_count",
                private_fork_next_event_cur_count, 0);
  if (private_fork_require_readonly_journal &&
      (!std::isfinite(private_fork_completed_event_time_sec) ||
       (private_fork_next_event_type != VIO &&
        private_fork_next_event_type != LIO) ||
       !std::isfinite(private_fork_next_event_absolute_time_sec) ||
       private_fork_next_event_cur_hash.empty() ||
       private_fork_next_event_image_hash.empty() ||
       private_fork_next_event_imu_hash.empty() ||
       private_fork_next_event_cur_count <= 0))
    throw std::invalid_argument(
        "private fork import requires the sealed next event identity");
  if (private_raw_scan_oracle_config.enabled &&
      (!private_fork_require_readonly_journal || !raw_lidar_reprocess_enabled ||
       estimated_range_scale_candidate_enabled || replay_range_scale_oracle_config.enabled))
    throw std::invalid_argument(
        "raw scan oracle requires a private readonly raw replay without other scale models");
  nh.param<bool>("diagnostic/private_lidar_profile_enabled",
                 private_lidar_profile_enabled, false);
  nh.param<std::string>("diagnostic/private_lidar_profile_path",
                        private_lidar_profile_path, std::string());
  if (private_lidar_profile_enabled)
  {
    if (!private_fork_require_readonly_journal || !raw_lidar_reprocess_enabled ||
        estimated_range_scale_candidate_enabled || replay_range_scale_oracle_config.enabled ||
        private_raw_scan_oracle_config.enabled || private_lidar_profile_path.empty())
      throw std::invalid_argument(
          "private LiDAR profile requires a readonly raw fork without other scale models");
    private_lidar_profile.load(private_lidar_profile_path);
  }
  if (private_fork_bind_journal && private_fork_require_readonly_journal)
    throw std::invalid_argument(
        "private fork export and import are mutually exclusive");
  nh.param<bool>("diagnostic/typed_event_map_snapshot_enabled",
                 typed_event_map_snapshot_enabled, false);
  nh.param<std::string>("diagnostic/typed_event_map_snapshot_directory",
                        typed_event_map_snapshot_directory, std::string());
  nh.param<std::string>("diagnostic/typed_event_map_snapshot_event_type",
                        typed_event_map_snapshot_event_type, std::string());
  nh.param<double>("diagnostic/typed_event_map_snapshot_relative_time_sec",
                   typed_event_map_snapshot_relative_time,
                   std::numeric_limits<double>::quiet_NaN());
  nh.param<std::string>("diagnostic/typed_event_map_snapshot_branch_role",
                        typed_event_map_snapshot_branch_role, std::string());
  nh.param<double>(
      "diagnostic/typed_event_map_snapshot_checkpoint_completed_relative_sec",
      typed_event_map_snapshot_checkpoint_completed,
      std::numeric_limits<double>::quiet_NaN());
  nh.param<std::string>("diagnostic/typed_event_map_snapshot_journal_sha256",
                        typed_event_map_snapshot_journal_sha256, std::string());
  if (typed_event_map_snapshot_enabled)
  {
    if (typed_event_map_snapshot_directory.empty())
      throw std::invalid_argument("typed event snapshot requires a directory");
    if (typed_event_map_snapshot_event_type != "LIO" &&
        typed_event_map_snapshot_event_type != "VIO")
      throw std::invalid_argument("typed event snapshot type must be LIO or VIO");
    if (!std::isfinite(typed_event_map_snapshot_relative_time))
      throw std::invalid_argument("typed event snapshot relative time is missing");
    if (typed_event_map_snapshot_branch_role != "direct" &&
        typed_event_map_snapshot_branch_role != "identity_replay")
      throw std::invalid_argument("typed event snapshot role must be direct or identity_replay");
    if (!rejected_candidate_map_export_directory.empty() &&
        typed_event_map_snapshot_directory.find(
            rejected_candidate_map_export_directory) == 0)
      throw std::invalid_argument(
          "typed event snapshot directory aliases rejected-map export");
    if (typed_event_map_snapshot_branch_role == "direct" &&
        !std::isfinite(typed_event_map_snapshot_checkpoint_completed))
      throw std::invalid_argument(
          "direct typed snapshot requires the checkpoint completed time");
    if (typed_event_map_snapshot_branch_role == "identity_replay")
    {
      if (!fast_livo::source_revision::validSha256(
              typed_event_map_snapshot_journal_sha256) ||
          private_fork_sealed_journal_path.empty())
        throw std::invalid_argument(
            "identity typed snapshot requires the sealed journal hash");
      if (fast_livo::source_revision::sha256File(
              private_fork_sealed_journal_path) !=
          typed_event_map_snapshot_journal_sha256)
        throw std::invalid_argument("sealed journal hash does not match the snapshot");
    }
  }
  replay_payload_journal.setPath(
      disk_backed_replay_payload_enabled ? replay_payload_journal_path : std::string());
  replay_payload_journal.configureAsync(async_replay_payload_spill_enabled,
                                        async_replay_payload_queue_bytes);
  if (raw_lidar_reprocess_enabled &&
      !disk_backed_replay_payload_enabled && rolling_history_enabled)
    ROS_WARN("[ RawReplay ] raw messages will remain in memory because disk-backed replay is disabled");
  nh.param<std::string>("trajectory/journal_path", trajectory_journal_path, std::string());
  nh.param<std::string>("shadow/abort_topic", shadow_abort_topic, std::string());
  nh.param<bool>("shadow/rearm_after_internal_rejection",
                 shadow_rearm_after_internal_rejection, false);
  nh.param<bool>("transactional_mapping/selective_lidar_gate_enabled", selective_lidar_gate_enabled, false);
  nh.param<double>("selective_lidar_gate/mad_multiplier",
                   selective_lidar_gate_config.mad_multiplier, 2.5);
  nh.param<double>("selective_lidar_gate/minimum_absolute_gate_m",
                   selective_lidar_gate_config.minimum_absolute_gate_m, 0.015);
  nh.param<double>("selective_lidar_gate/maximum_absolute_gate_m",
                   selective_lidar_gate_config.maximum_absolute_gate_m, 0.15);
  nh.param<double>("selective_lidar_gate/minimum_retained_fraction",
                   selective_lidar_gate_config.minimum_retained_fraction, 0.60);
  selective_lidar_gate_config.validate();
  nh.param<bool>("transactional_mapping/robust_lidar_weight_enabled",
                 robust_lidar_weight_enabled, false);
  nh.param<double>("robust_lidar_weight/mad_multiplier",
                   robust_lidar_weight_config.mad_multiplier, 2.5);
  nh.param<double>("robust_lidar_weight/minimum_transition_m",
                   robust_lidar_weight_config.minimum_transition_m, 0.015);
  nh.param<double>("robust_lidar_weight/maximum_transition_m",
                   robust_lidar_weight_config.maximum_transition_m, 0.075);
  nh.param<double>("robust_lidar_weight/minimum_weight",
                   robust_lidar_weight_config.minimum_weight, 0.10);
  robust_lidar_weight_config.validate();
  if (selective_lidar_gate_enabled && robust_lidar_weight_enabled)
    throw std::invalid_argument(
        "selective hard gating and robust residual weighting are mutually exclusive");
  nh.param<bool>("transactional_mapping/rolling_history_enabled", rolling_history_enabled, false);
  nh.param<double>("transactional_mapping/rolling_checkpoint_period_sec", rolling_checkpoint_period_sec, 5.0);
  nh.param<double>("transactional_mapping/rolling_history_horizon_sec", rolling_history_horizon_sec, 20.0);
  int rolling_max_checkpoints_param = static_cast<int>(rolling_max_checkpoints);
  nh.param<int>("transactional_mapping/rolling_max_checkpoints", rolling_max_checkpoints_param, rolling_max_checkpoints_param);
  rolling_max_checkpoints = static_cast<std::size_t>(std::max(1, rolling_max_checkpoints_param));
  int rolling_delta_max_checkpoints_param =
      static_cast<int>(rolling_delta_max_checkpoints);
  nh.param<int>("transactional_mapping/rolling_delta_max_checkpoints",
                rolling_delta_max_checkpoints_param,
                rolling_delta_max_checkpoints_param);
  rolling_delta_max_checkpoints = static_cast<std::size_t>(
      std::max(1, rolling_delta_max_checkpoints_param));
  nh.param<bool>("transactional_mapping/rolling_preserve_horizon_anchor",
                 rolling_preserve_horizon_anchor, false);
  nh.param<double>("transactional_mapping/handoff_maximum_replay_gap_sec",
                   recovery_handoff_maximum_replay_gap_sec, 0.25);
  nh.param<std::string>("transactional_mapping/handoff_export_directory",
                         recovery_handoff_export_directory, std::string());
  nh.param<std::string>("transactional_mapping/handoff_import_directory",
                         recovery_handoff_import_directory, std::string());
  nh.param<bool>("transactional_mapping/handoff_stop_export_after_consume",
                 recovery_handoff_stop_export_after_consume, false);
  if (rolling_checkpoint_period_sec <= 0.0 || rolling_history_horizon_sec <= 0.0)
  {
    throw std::invalid_argument("rolling checkpoint period and history horizon must be positive");
  }
  if (!std::isfinite(recovery_handoff_maximum_replay_gap_sec) ||
      recovery_handoff_maximum_replay_gap_sec <= 0.0)
  {
    throw std::invalid_argument(
        "handoff maximum replay gap must be finite and positive");
  }
  nh.param<bool>("harm_detector/enabled", online_harm_detector_enabled, false);
  nh.param<bool>("harm_detector/observe_only", harm_detector_observe_only, false);
  nh.param<bool>("incremental_public_guard/enabled",
                 incremental_guard_config.enabled, false);
  nh.param<double>("incremental_public_guard/minimum_onset_confidence",
                   incremental_guard_config.minimum_onset_confidence, 0.60);
  nh.param<double>("incremental_public_guard/horizon_sec",
                   incremental_guard_config.horizon_sec, 8.0);
  nh.param<double>("incremental_public_guard/lidar_information_weight",
                   incremental_guard_config.lidar_information_weight, 0.10);
  nh.param<double>("incremental_public_guard/terminal_range_scale",
                   incremental_guard_config.terminal_range_scale, 1.04);
  int incremental_probe_matches = 80;
  int incremental_probe_passes = 3;
  nh.param<int>("incremental_public_guard/minimum_probe_matches",
                incremental_probe_matches, 80);
  nh.param<int>("incremental_public_guard/consecutive_probe_passes",
                incremental_probe_passes, 3);
  if (incremental_probe_matches <= 0 || incremental_probe_passes <= 0)
    throw std::invalid_argument("invalid incremental scale-probe support");
  incremental_guard_config.minimum_probe_matches =
      static_cast<std::size_t>(incremental_probe_matches);
  incremental_guard_config.consecutive_probe_passes =
      static_cast<std::size_t>(incremental_probe_passes);
  nh.param<double>("incremental_public_guard/minimum_probe_relative_improvement",
                   incremental_guard_config.minimum_probe_relative_improvement, 0.20);
  nh.param<double>("incremental_public_guard/minimum_probe_absolute_improvement_m",
                   incremental_guard_config.minimum_probe_absolute_improvement_m, 0.004);
  if (incremental_guard_config.enabled &&
      (!online_harm_detector_enabled || !harm_detector_observe_only ||
       ros::this_node::getNamespace() != "/shadow/incumbent"))
    throw std::invalid_argument(
        "incremental public guard requires the observe-only incumbent detector");
  incremental_public_guard.configure(incremental_guard_config);
  incremental_guard_evidence.configure(incremental_guard_config);
  nh.param<bool>("diagnostic/private_frozen_scale_probe_enabled",
                 private_frozen_scale_probe_enabled, false);
  nh.param<bool>("diagnostic/private_frozen_scale_probe_snapshot_reference_enabled",
                 private_frozen_scale_probe_snapshot_reference_enabled, false);
  nh.param<bool>("diagnostic/private_frozen_scale_probe_pose_refinement_enabled",
                 private_frozen_scale_probe_pose_refinement_enabled, false);
  nh.param<bool>("diagnostic/private_frozen_scale_probe_common_support_enabled",
                 private_frozen_scale_probe_common_support_enabled, false);
  nh.param<double>("diagnostic/private_frozen_scale_probe_reference_sec",
                   private_frozen_scale_probe_reference_sec, 45.0);
  nh.param<double>("diagnostic/private_frozen_scale_probe_period_sec",
                   private_frozen_scale_probe_period_sec, 0.7);
  nh.param<double>("diagnostic/private_frozen_scale_probe_end_sec",
                   private_frozen_scale_probe_end_sec, 65.0);
  std::string private_frozen_scale_probe_path;
  nh.param<std::string>("diagnostic/private_frozen_scale_probe_path",
                        private_frozen_scale_probe_path, std::string());
  if (private_frozen_scale_probe_enabled)
  {
    if (private_frozen_scale_probe_pose_refinement_enabled &&
        !private_frozen_scale_probe_snapshot_reference_enabled)
      throw std::invalid_argument(
          "private pose refinement requires immutable snapshot reference");
    if (private_frozen_scale_probe_common_support_enabled &&
        !private_frozen_scale_probe_pose_refinement_enabled)
      throw std::invalid_argument(
          "private common support requires pose refinement");
    if (!raw_lidar_reprocess_enabled ||
        !std::isfinite(private_frozen_scale_probe_reference_sec) ||
        private_frozen_scale_probe_reference_sec < 0.0 ||
        !std::isfinite(private_frozen_scale_probe_period_sec) ||
        private_frozen_scale_probe_period_sec <= 0.0 ||
        !std::isfinite(private_frozen_scale_probe_end_sec) ||
        private_frozen_scale_probe_end_sec <=
            private_frozen_scale_probe_reference_sec ||
        private_frozen_scale_probe_path.empty())
      throw std::invalid_argument(
          "private frozen scale probe requires raw replay, valid timing and CSV path");
    private_frozen_scale_probe_csv.open(private_frozen_scale_probe_path,
                                       std::ios::out | std::ios::trunc);
    if (!private_frozen_scale_probe_csv.is_open())
      throw std::runtime_error("cannot open private frozen scale probe CSV");
    private_frozen_scale_probe_csv
        << "reference_sec,event_sec,scale,raw_frames,corrected_points,"
           "downsampled_points,matched_points,match_ratio,mean_abs_residual_m,"
           "identity_cloud_equal,source_cloud_digest,probe_cloud_digest,"
           "probe_state_x,probe_state_y,probe_state_z,status,"
           "pose_refinement_steps,initial_matched_points,"
           "initial_mean_abs_residual_m,pose_shift_m,"
           "common_shared_points,common_reference_matches,"
           "common_reference_residual_m,common_candidate_residual_m,"
           "common_world_displacement_m,common_status\n";
    private_frozen_scale_probe_csv << std::setprecision(17);
  }
  nh.param<bool>("harm_detector/frozen_event_enabled",
                 frozen_detection_event.enabled, false);
  nh.param<double>("harm_detector/frozen_detection_sec",
                   frozen_detection_event.detection_sec, -1.0);
  nh.param<double>("harm_detector/frozen_onset_lower_sec",
                   frozen_detection_event.lower_sec, -1.0);
  nh.param<double>("harm_detector/frozen_onset_point_sec",
                   frozen_detection_event.point_sec, -1.0);
  nh.param<double>("harm_detector/frozen_onset_upper_sec",
                   frozen_detection_event.upper_sec, -1.0);
  nh.param<double>("harm_detector/frozen_onset_confidence",
                   frozen_detection_event.confidence, -1.0);
  int frozen_episode_id = 0;
  int frozen_proposal_index = 0;
  nh.param<int>("harm_detector/frozen_episode_id", frozen_episode_id, 0);
  nh.param<int>("harm_detector/frozen_proposal_index_in_episode",
                frozen_proposal_index, 0);
  frozen_detection_event.episode_id =
      static_cast<std::size_t>(std::max(0, frozen_episode_id));
  frozen_detection_event.proposal_index_in_episode =
      static_cast<std::size_t>(std::max(0, frozen_proposal_index));
  nh.param<std::string>("harm_detector/frozen_source_manifest_sha256",
                         frozen_detection_event.source_manifest_sha256,
                         std::string());
  nh.param<std::string>("harm_detector/diagnostics_path",
                        harm_detector_diagnostics_path, std::string());
  harm_detector_config.enabled = online_harm_detector_enabled;
  nh.param<bool>("harm_detector/two_timescale_enabled",
                 harm_detector_config.two_timescale_enabled, false);
  nh.param<double>("harm_detector/arming_delay_sec",
                   harm_detector_config.arming_delay_sec, 0.0);
  int baseline_window = static_cast<int>(harm_detector_config.baseline_window);
  int trigger_consecutive = static_cast<int>(harm_detector_config.trigger_consecutive);
  nh.param<int>("harm_detector/baseline_window", baseline_window, baseline_window);
  nh.param<int>("harm_detector/trigger_consecutive", trigger_consecutive, trigger_consecutive);
  harm_detector_config.baseline_window = static_cast<std::size_t>(std::max(10, baseline_window));
  harm_detector_config.trigger_consecutive = static_cast<std::size_t>(std::max(1, trigger_consecutive));
  nh.param<double>("harm_detector/minimum_anomaly_duration_sec",
                   harm_detector_config.minimum_anomaly_duration_sec, 0.5);
  nh.param<double>("harm_detector/residual_mad_multiplier", harm_detector_config.residual_mad_multiplier, 6.0);
  nh.param<double>("harm_detector/residual_increase_floor", harm_detector_config.residual_increase_floor, 0.003);
  nh.param<double>("harm_detector/match_mad_multiplier", harm_detector_config.match_mad_multiplier, 6.0);
  nh.param<double>("harm_detector/match_drop_floor", harm_detector_config.match_drop_floor, 0.10);
  nh.param<double>("harm_detector/innovation_mad_multiplier",
                   harm_detector_config.innovation_mad_multiplier, 6.0);
  nh.param<double>("harm_detector/innovation_increase_floor",
                   harm_detector_config.innovation_increase_floor, 0.50);
  nh.param<double>("harm_detector/residual_weight",
                   harm_detector_config.residual_weight, 1.0);
  nh.param<double>("harm_detector/match_weight",
                   harm_detector_config.match_weight, 1.0);
  nh.param<double>("harm_detector/innovation_weight",
                   harm_detector_config.innovation_weight, 1.0);
  nh.param<double>("harm_detector/strong_fused_score",
                   harm_detector_config.strong_fused_score, 0.75);
  int minimum_strong_signals =
      static_cast<int>(harm_detector_config.minimum_strong_signals);
  nh.param<int>("harm_detector/minimum_strong_signals", minimum_strong_signals,
                minimum_strong_signals);
  harm_detector_config.minimum_strong_signals =
      static_cast<std::size_t>(std::max(1, std::min(3, minimum_strong_signals)));
  nh.param<double>("harm_detector/dominant_signal_score",
                   harm_detector_config.dominant_signal_score, 2.5);
  int onset_guard_frames = static_cast<int>(harm_detector_config.onset_guard_frames);
  nh.param<int>("harm_detector/onset_guard_frames", onset_guard_frames, onset_guard_frames);
  harm_detector_config.onset_guard_frames = static_cast<std::size_t>(std::max(1, onset_guard_frames));
  nh.param<double>("harm_detector/onset_min_guard_sec", harm_detector_config.onset_min_guard_sec, 0.5);
  nh.param<double>("harm_detector/onset_max_guard_sec", harm_detector_config.onset_max_guard_sec, 2.0);
  nh.param<double>("harm_detector/weak_residual_fraction",
                   harm_detector_config.weak_residual_fraction, 0.35);
  nh.param<double>("harm_detector/weak_match_fraction",
                   harm_detector_config.weak_match_fraction, 0.35);
  nh.param<double>("harm_detector/weak_innovation_fraction",
                   harm_detector_config.weak_innovation_fraction, 0.35);
  nh.param<double>("harm_detector/weak_combined_score",
                   harm_detector_config.weak_combined_score, 0.75);
  int weak_max_gap_frames = static_cast<int>(harm_detector_config.weak_max_gap_frames);
  nh.param<int>("harm_detector/weak_max_gap_frames", weak_max_gap_frames,
                weak_max_gap_frames);
  harm_detector_config.weak_max_gap_frames =
      static_cast<std::size_t>(std::max(0, weak_max_gap_frames));
  nh.param<double>("harm_detector/weak_lookback_sec",
                   harm_detector_config.weak_lookback_sec, 2.5);
  nh.param<double>("harm_detector/episode_merge_gap_sec",
                   harm_detector_config.episode_merge_gap_sec, 12.0);
  nh.param<double>("harm_detector/reference_window_sec",
                   harm_detector_config.reference_window_sec, 8.0);
  nh.param<double>("harm_detector/reference_gap_sec",
                   harm_detector_config.reference_gap_sec, 0.5);
  nh.param<double>("harm_detector/current_window_sec",
                   harm_detector_config.current_window_sec, 2.0);
  nh.param<bool>("harm_detector/two_timescale_innovation_two_sided",
                 harm_detector_config.two_timescale_innovation_two_sided, true);
  nh.param<double>("harm_detector/two_timescale_minimum_residual_anchor_score",
                   harm_detector_config.two_timescale_minimum_residual_anchor_score,
                   0.35);
  nh.param<double>("harm_detector/two_timescale_signal_support_score",
                   harm_detector_config.two_timescale_signal_support_score, 1.0);
  nh.param<bool>("harm_detector/two_timescale_conservative_weak_onset_enabled",
                 harm_detector_config.two_timescale_conservative_weak_onset_enabled,
                 false);
  nh.param<double>(
      "harm_detector/two_timescale_conservative_weak_score_threshold",
      harm_detector_config.two_timescale_conservative_weak_score_threshold,
      1.0);
  nh.param<double>("harm_detector/quarantine_duration_sec", online_quarantine_duration_sec, 8.0);
  nh.param<bool>("recovery_proposal_arbiter/enabled",
                 recovery_proposal_arbiter_config.enabled, false);
  int maximum_attempts_per_episode = static_cast<int>(
      recovery_proposal_arbiter_config.maximum_attempts_per_episode);
  nh.param<int>("recovery_proposal_arbiter/maximum_attempts_per_episode",
                maximum_attempts_per_episode, maximum_attempts_per_episode);
  recovery_proposal_arbiter_config.maximum_attempts_per_episode =
      static_cast<std::size_t>(std::max(0, maximum_attempts_per_episode));
  int maximum_attempts_total = static_cast<int>(
      recovery_proposal_arbiter_config.maximum_attempts_total);
  nh.param<int>("recovery_proposal_arbiter/maximum_attempts_total",
                maximum_attempts_total, maximum_attempts_total);
  recovery_proposal_arbiter_config.maximum_attempts_total =
      static_cast<std::size_t>(std::max(0, maximum_attempts_total));
  int maximum_rejected_attempt_refunds_total = static_cast<int>(
      recovery_proposal_arbiter_config.maximum_rejected_attempt_refunds_total);
  nh.param<int>(
      "recovery_proposal_arbiter/maximum_rejected_attempt_refunds_total",
      maximum_rejected_attempt_refunds_total,
      maximum_rejected_attempt_refunds_total);
  recovery_proposal_arbiter_config.maximum_rejected_attempt_refunds_total =
      static_cast<std::size_t>(
          std::max(0, maximum_rejected_attempt_refunds_total));
  nh.param<double>("recovery_proposal_arbiter/minimum_residual_score",
                   recovery_proposal_arbiter_config.minimum_residual_score,
                   0.0);
  nh.param<double>("recovery_proposal_arbiter/minimum_raw_residual",
                   recovery_proposal_arbiter_config.minimum_raw_residual,
                   0.0);
  nh.param<bool>(
      "recovery_proposal_arbiter/strong_admission_first_proposal_only",
      recovery_proposal_arbiter_config.strong_admission_first_proposal_only,
      false);
  nh.param<bool>(
      "recovery_proposal_arbiter/confirmed_lidar_fallback_enabled",
      recovery_proposal_arbiter_config.confirmed_lidar_fallback_enabled,
      false);
  int confirmed_lidar_fallback_minimum_proposals = static_cast<int>(
      recovery_proposal_arbiter_config
          .confirmed_lidar_fallback_minimum_proposals);
  nh.param<int>(
      "recovery_proposal_arbiter/confirmed_lidar_fallback_minimum_proposals",
      confirmed_lidar_fallback_minimum_proposals,
      confirmed_lidar_fallback_minimum_proposals);
  recovery_proposal_arbiter_config
      .confirmed_lidar_fallback_minimum_proposals =
          static_cast<std::size_t>(
              std::max(0, confirmed_lidar_fallback_minimum_proposals));
  nh.param<double>(
      "recovery_proposal_arbiter/"
      "confirmed_lidar_fallback_minimum_confirmation_raw_residual",
      recovery_proposal_arbiter_config
          .confirmed_lidar_fallback_minimum_confirmation_raw_residual,
      0.0);
  nh.param<double>(
      "recovery_proposal_arbiter/"
      "confirmed_lidar_fallback_maximum_innovation_to_lidar_ratio",
      recovery_proposal_arbiter_config
          .confirmed_lidar_fallback_maximum_innovation_to_lidar_ratio,
      1.5);
  nh.param<bool>(
      "recovery_proposal_arbiter/require_external_corroboration",
      recovery_proposal_arbiter_config.require_external_corroboration,
      false);
  nh.param<std::string>(
      "recovery_proposal_arbiter/external_corroboration_topic",
      recovery_proposal_corroboration_topic, std::string());
  nh.param<double>(
      "recovery_proposal_arbiter/external_corroboration_tolerance_sec",
      recovery_proposal_corroboration_tolerance_sec, 12.0);
  nh.param<double>(
      "recovery_proposal_arbiter/minimum_signal_escalation_ratio",
      recovery_proposal_arbiter_config.minimum_signal_escalation_ratio, 0.0);
  nh.param<bool>(
      "recovery_proposal_arbiter/rejected_history_use_raw_residual",
      recovery_proposal_arbiter_config.rejected_history_use_raw_residual,
      false);
  recovery_proposal_arbiter_config.validate();
  if (recovery_proposal_arbiter_config.require_external_corroboration &&
      recovery_proposal_corroboration_topic.empty())
    throw std::invalid_argument(
        "external recovery corroboration requires a proposal topic");
  if (!std::isfinite(recovery_proposal_corroboration_tolerance_sec) ||
      recovery_proposal_corroboration_tolerance_sec <= 0.0)
    throw std::invalid_argument(
        "external recovery corroboration tolerance must be positive");

  nh.param<bool>("healthy_tail_validator/enabled", healthy_tail_config.enabled,
                 online_harm_detector_enabled);
  int consecutive_healthy = static_cast<int>(healthy_tail_config.consecutive_healthy);
  nh.param<int>("healthy_tail_validator/consecutive_healthy", consecutive_healthy,
                consecutive_healthy);
  healthy_tail_config.consecutive_healthy =
      static_cast<std::size_t>(std::max(1, consecutive_healthy));
  nh.param<double>("healthy_tail_validator/minimum_duration_sec",
                   healthy_tail_config.minimum_healthy_duration_sec, 1.5);
  nh.param<double>("healthy_tail_validator/timeout_sec",
                   healthy_tail_config.timeout_sec, 12.0);
  nh.param<bool>("healthy_tail_validator/sequential_evidence_enabled",
                 healthy_tail_config.sequential_evidence_enabled, false);
  int minimum_sequential_observations = static_cast<int>(
      healthy_tail_config.minimum_sequential_observations);
  nh.param<int>("healthy_tail_validator/minimum_sequential_observations",
                minimum_sequential_observations,
                minimum_sequential_observations);
  healthy_tail_config.minimum_sequential_observations =
      static_cast<std::size_t>(std::max(1, minimum_sequential_observations));
  nh.param<double>("healthy_tail_validator/confidence_z",
                   healthy_tail_config.confidence_z, 1.645);
  nh.param<double>(
      "healthy_tail_validator/minimum_residual_pass_rate_lower_bound",
      healthy_tail_config.minimum_residual_pass_rate_lower_bound, 0.60);
  nh.param<double>(
      "healthy_tail_validator/minimum_match_pass_rate_lower_bound",
      healthy_tail_config.minimum_match_pass_rate_lower_bound, 0.60);
  nh.param<double>(
      "healthy_tail_validator/minimum_joint_pass_rate_lower_bound",
      healthy_tail_config.minimum_joint_pass_rate_lower_bound, 0.55);
  nh.param<bool>("healthy_tail_validator/commit_dry_run",
                 healthy_tail_commit_dry_run, false);
  nh.param<bool>("healthy_tail_validator/candidate_summary_required",
                 healthy_tail_config.candidate_summary_required, false);
  nh.param<double>("healthy_tail_validator/maximum_reentry_position_correction_m",
                   healthy_tail_config.maximum_reentry_position_correction_m, 0.50);
  nh.param<double>("healthy_tail_validator/maximum_reentry_rotation_correction_deg",
                   healthy_tail_config.maximum_reentry_rotation_correction_deg, 5.0);
  nh.param<double>("healthy_tail_validator/minimum_map_coverage_ratio",
                   healthy_tail_config.minimum_map_coverage_ratio, 1.0);
  nh.param<double>("healthy_tail_validator/minimum_map_support_ratio",
                   healthy_tail_config.minimum_map_support_ratio, 0.75);
  nh.param<double>("healthy_tail_validator/maximum_boundary_position_divergence_m",
                   healthy_tail_config.maximum_boundary_position_divergence_m, 1.0);
  nh.param<double>("healthy_tail_validator/maximum_boundary_rotation_divergence_deg",
                   healthy_tail_config.maximum_boundary_rotation_divergence_deg, 5.0);
  nh.param<bool>("healthy_tail_validator/use_residual_check",
                 healthy_tail_config.use_residual_check, true);
  nh.param<bool>("healthy_tail_validator/use_match_ratio_check",
                 healthy_tail_config.use_match_ratio_check, true);
  nh.param<bool>("healthy_tail_validator/use_reentry_position_check",
                 healthy_tail_config.use_reentry_position_check, true);
  nh.param<bool>("healthy_tail_validator/use_reentry_rotation_check",
                 healthy_tail_config.use_reentry_rotation_check, true);
  nh.param<bool>("healthy_tail_validator/use_map_coverage_check",
                 healthy_tail_config.use_map_coverage_check, true);
  nh.param<bool>("healthy_tail_validator/use_map_support_check",
                 healthy_tail_config.use_map_support_check, true);
  nh.param<bool>("healthy_tail_validator/use_boundary_position_check",
                 healthy_tail_config.use_boundary_position_check, true);
  nh.param<bool>("healthy_tail_validator/use_boundary_rotation_check",
                 healthy_tail_config.use_boundary_rotation_check, true);
  int healthy_baseline_window_param = static_cast<int>(healthy_baseline_window);
  nh.param<int>("healthy_tail_validator/baseline_window", healthy_baseline_window_param,
                healthy_baseline_window_param);
  healthy_baseline_window = static_cast<std::size_t>(std::max(20, healthy_baseline_window_param));
  healthy_tail_config.validate();
  nh.param<bool>("recovery_benefit_validator/enabled",
                 recovery_benefit_config.enabled, false);
  int benefit_minimum_observations = static_cast<int>(
      recovery_benefit_config.minimum_observations);
  nh.param<int>("recovery_benefit_validator/minimum_observations",
                benefit_minimum_observations,
                benefit_minimum_observations);
  recovery_benefit_config.minimum_observations =
      static_cast<std::size_t>(std::max(1, benefit_minimum_observations));
  nh.param<double>("recovery_benefit_validator/confidence_z",
                   recovery_benefit_config.confidence_z, 1.645);
  nh.param<double>("recovery_benefit_validator/residual_relative_improvement",
                   recovery_benefit_config.residual_relative_improvement, 0.05);
  nh.param<double>("recovery_benefit_validator/match_absolute_improvement",
                   recovery_benefit_config.match_absolute_improvement, 0.02);
  nh.param<double>("recovery_benefit_validator/innovation_relative_improvement",
                   recovery_benefit_config.innovation_relative_improvement, 0.30);
  nh.param<double>(
      "recovery_benefit_validator/minimum_improved_fraction_lower_bound",
      recovery_benefit_config.minimum_improved_fraction_lower_bound, 0.40);
  nh.param<bool>(
      "recovery_benefit_validator/allow_absolute_health_dominance",
      recovery_benefit_allow_absolute_health_dominance, false);
  nh.param<bool>(
      "recovery_benefit_validator/common_reference_map_enabled",
      recovery_benefit_common_reference_map_enabled, false);
  recovery_benefit_config.validate();
  nh.param<double>("healthy_tail_validator/residual_mad_multiplier",
                   healthy_residual_mad_multiplier, 3.0);
  nh.param<double>("healthy_tail_validator/residual_increase_floor",
                   healthy_residual_increase_floor, 0.0015);
  nh.param<double>("healthy_tail_validator/match_mad_multiplier",
                   healthy_match_mad_multiplier, 3.0);
  nh.param<double>("healthy_tail_validator/match_drop_floor",
                   healthy_match_drop_floor, 0.05);
  if (online_harm_detector_enabled && !harm_detector_observe_only &&
      (!rolling_history_enabled || !map_transaction_config.enabled))
    throw std::invalid_argument("online harm detector requires enabled transactional rolling history");
  if (frozen_detection_event.enabled &&
      (!online_harm_detector_enabled || harm_detector_observe_only ||
       recovery_handoff_import_directory.empty() ||
       !std::isfinite(frozen_detection_event.detection_sec) ||
       !std::isfinite(frozen_detection_event.lower_sec) ||
       !std::isfinite(frozen_detection_event.point_sec) ||
       !std::isfinite(frozen_detection_event.upper_sec) ||
       !std::isfinite(frozen_detection_event.confidence) ||
       frozen_detection_event.lower_sec < 0.0 ||
       frozen_detection_event.lower_sec > frozen_detection_event.point_sec ||
       frozen_detection_event.point_sec > frozen_detection_event.upper_sec ||
       frozen_detection_event.upper_sec > frozen_detection_event.detection_sec ||
       frozen_detection_event.confidence < 0.0 ||
       frozen_detection_event.confidence > 1.0 ||
       frozen_detection_event.episode_id == 0 ||
       frozen_detection_event.proposal_index_in_episode == 0 ||
       frozen_detection_event.source_manifest_sha256.size() != 64))
    throw std::invalid_argument("invalid frozen detector event or missing sealed handoff import");
  if (estimated_range_scale_candidate_enabled &&
      (!online_harm_detector_enabled || harm_detector_observe_only))
    throw std::invalid_argument(
        "estimated range-scale candidate requires an active online harm detector");
  if (healthy_tail_commit_dry_run && shadow_abort_topic.empty())
    throw std::invalid_argument(
        "healthy-tail commit dry-run requires a process-isolated shadow candidate");
  if (shadow_rearm_after_internal_rejection &&
      (shadow_abort_topic.empty() || !online_harm_detector_enabled ||
       !rolling_history_enabled || !delta_voxel_journal_enabled))
    throw std::invalid_argument(
        "shadow candidate re-arm requires process isolation, online rolling history and delta journal");
  if (online_quarantine_duration_sec <= 0.0)
    throw std::invalid_argument("online detector quarantine duration must be positive");
  if (healthy_residual_mad_multiplier < 0.0 || healthy_residual_increase_floor <= 0.0 ||
      healthy_match_mad_multiplier < 0.0 || healthy_match_drop_floor <= 0.0)
    throw std::invalid_argument("healthy-tail evidence thresholds must be positive");

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  const std::string camera_parameter_namespace = ros::this_node::getName();
  if (!vk::camera_loader::loadFromRosNs(camera_parameter_namespace,
                                        vio_manager->cam))
    throw std::runtime_error("Camera model not correctly specified in " +
                             camera_parameter_namespace);

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->patch_boundary_check_only = vio_patch_boundary_check_only;
  vio_manager->patch_boundary_guard = vio_patch_boundary_guard;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->initializeVIO();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() 
{
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_en)
  {
    const std::string lidar_pose_path = joinPath(pcd_output_directory, "lidar_poses.txt");
    fout_lidar_pos.open(lidar_pose_path, std::ios::out);
    if (!fout_lidar_pos.is_open())
      ROS_ERROR_STREAM("Cannot open LiDAR pose output: " << lidar_pose_path);
  }
  if(img_save_en) fout_visual_pos.open(std::string(ROOT_DIR) + "Log/image/image_poses.txt", std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it) 
{
  sub_pcl = p_pre->lidar_type == AVIA ? 
            nh.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this): 
            nh.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);
  sub_img = nh.subscribe(img_topic, 200000, &LIVMapper::img_cbk, this);
  
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
  transaction_status_pub = nh.advertise<std_msgs::String>("/transaction_status", 10, true);
  if (!shadow_abort_topic.empty())
    shadow_abort_sub = nh.subscribe(shadow_abort_topic, 1,
                                    &LIVMapper::shadowAbortCallback, this);
  if (!recovery_proposal_corroboration_topic.empty())
    recovery_proposal_corroboration_sub = nh.subscribe(
        recovery_proposal_corroboration_topic, 20,
        &LIVMapper::recoveryProposalCorroborationCallback, this);
  source_revision_barrier_enabled = sourceRevisionBarrierEnabled() &&
      ros::this_node::getNamespace() == "/shadow/candidate";
  source_revision_payload_hash_enabled = sourceRevisionPayloadHashEnabled() &&
      ros::this_node::getNamespace() == "/shadow/candidate";
  if (source_revision_payload_hash_enabled && !source_revision_barrier_enabled)
    throw std::invalid_argument(
        "source revision payload hash requires prepublic barrier");
  source_revision_cut_enabled = (sourceRevisionCutEnabled() ||
      source_revision_barrier_enabled) &&
      ros::this_node::getNamespace() == "/shadow/candidate";
  if (source_revision_cut_enabled)
  {
    source_revision_cut_request_sub = nh.subscribe(
        "/shadow/candidate/source_revision_cut_request", 1,
        &LIVMapper::sourceRevisionCutRequestCallback, this);
    if (source_revision_barrier_enabled)
      source_revision_cut_ready_pub = nh.advertise<std_msgs::String>(
          "/shadow/candidate/source_revision_cut_ready", 1, true);
  }
  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  nh.setParam("diagnostic/mapper_subscribers_ready", true);
  ROS_INFO("[ Mapper ] subscribers ready");
  pubImage = it.advertise("/rgb_img", 1);
  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
  publishTransactionStatus("monitoring");
}

void LIVMapper::publishTransactionStatus(const std::string &state,
                                         const std::string &detail,
                                         const double relative_time_sec)
{
  if (p2_runtime_receipt.enabled())
  {
    const std::uint64_t now = fast_livo::p2_receipt::monotonicNs();
    if (state == "replay_started")
    {
      p2_replay_begin_ns = now;
      p2_replay_end_ns = 0;
      p2_replay_consumed_events = 0;
      p2_replay_total_events = map_transaction_buffer.size();
    }
    if (state == "replay_completed" || state == "replay_failed" ||
        state == "candidate_rejected" || state == "candidate_terminated" ||
        state == "rolled_back")
    {
      if (p2_replay_begin_ns != 0 && p2_replay_end_ns == 0)
        p2_replay_end_ns = now;
    }
    if (state == "candidate_rearmed") ++p2_rearm_count;
    fast_livo::p2_receipt::Row row;
    row.record_type = "status";
    row.route = "transaction";
    row.event_id = p2_current_event_id;
    row.stage = p2_current_route;
    row.modality = p2_current_modality;
    row.sensor_max_input_stamp_ns = p2_current_sensor_max_input_stamp_ns;
    row.input_stamp_semantics = "last_consumed_group_context_not_new_arrival";
    row.proposal_episode_id = p2_active_episode_id;
    row.proposal_index_in_episode = p2_active_proposal_index;
    row.proposal_origin = p2_active_proposal_origin;
    row.status = state;
    row.detail = detail;
    row.buffer_events = map_transaction_buffer.size();
    row.replay_consumed_events = p2_replay_consumed_events;
    row.replay_total_events = p2_replay_total_events;
    row.replay_elapsed_ns = p2_replay_begin_ns == 0 ? 0 :
        (p2_replay_end_ns == 0 ? now : p2_replay_end_ns) - p2_replay_begin_ns;
    row.rearm_count = p2_rearm_count;
    p2_runtime_receipt.write(row);
    const bool suppressed_without_pending_corroboration =
        state == "recovery_proposal_suppressed" &&
        detail.find("external_corroboration_pending") == std::string::npos;
    if (state == "candidate_rearmed" || state == "candidate_terminated" ||
        state == "candidate_committed" ||
        suppressed_without_pending_corroboration)
    {
      p2_active_episode_id = 0;
      p2_active_proposal_index = 0;
      p2_active_proposal_origin.clear();
    }
  }
  if (!transaction_status_pub) return;
  std_msgs::String message;
  message.data = state;
  if (!detail.empty()) message.data += ";detail=" + detail;
  if (std::isfinite(relative_time_sec))
  {
    std::ostringstream timing;
    timing << std::setprecision(17)
           << ";relative_time_sec=" << relative_time_sec;
    if (is_first_frame && std::isfinite(_first_lidar_time))
      timing << ";sensor_time_sec=" << (_first_lidar_time + relative_time_sec);
    message.data += timing.str();
  }
  transaction_status_pub.publish(message);
}

void LIVMapper::recordP2Fusion(const char *record_type, const char *route,
                               std::uint64_t event_id, const char *stage,
                               const LidarMeasureGroup &group)
{
  if (!p2_runtime_receipt.enabled()) return;
  if (std::string(record_type) == "consumed")
  {
    p2_current_route = route;
    p2_current_event_id = event_id;
    p2_current_modality = std::to_string(static_cast<int>(group.lio_vio_flg));
    p2_current_sensor_max_input_stamp_ns = p2SensorStampNs(group);
    p2_current_fusion_pending = true;
    if (p2_live_metric_receipt.enabled())
    {
      // Observation cache is deliberately not checkpointed. A VIO/no-update
      // event must never inherit the preceding LIO metric after a restore.
      p2_live_metric_cache = fast_livo::p2_live_metric::Metric{};
      p2_live_metric_route.clear();
      p2_live_metric_event_id = 0;
      p2_live_metric_consumed_ns = 0;
    }
  }
  else if (std::string(record_type) == "result_ready" ||
           std::string(record_type) == "disposition")
    p2_current_fusion_pending = false;
  fast_livo::p2_receipt::Row row;
  row.record_type = record_type;
  row.route = route;
  row.event_id = event_id;
  row.stage = stage;
  row.modality = p2_current_modality;
  // post-update may mutate scratch group; never recompute the consumed bound.
  row.sensor_max_input_stamp_ns = p2_current_sensor_max_input_stamp_ns;
  row.input_stamp_semantics =
      "steady_ns_post_sync_availability_upper_bound;max_stamp_sensor_clock;not_callback_arrival";
  row.proposal_episode_id = p2_active_episode_id;
  row.proposal_index_in_episode = p2_active_proposal_index;
  row.proposal_origin = p2_active_proposal_origin;
  row.buffer_events = map_transaction_buffer.size();
  row.replay_consumed_events = p2_replay_consumed_events;
  row.replay_total_events = p2_replay_total_events;
  row.replay_elapsed_ns = p2_replay_begin_ns == 0 ? 0 :
      (p2_replay_end_ns == 0 ? fast_livo::p2_receipt::monotonicNs() :
       p2_replay_end_ns) - p2_replay_begin_ns;
  row.rearm_count = p2_rearm_count;
  p2_runtime_receipt.write(row);
  if (p2_live_metric_receipt.enabled())
  {
    if (std::string(record_type) == "consumed")
      p2_live_metric_consumed_ns = fast_livo::p2_receipt::monotonicNs();
    else if (std::string(record_type) == "result_ready")
    {
      fast_livo::p2_live_metric::Row metric_row;
      metric_row.route = route;
      metric_row.event_id = event_id;
      metric_row.stage = stage;
      metric_row.modality = p2_current_modality;
      metric_row.sensor_max_input_stamp_ns = p2_current_sensor_max_input_stamp_ns;
      metric_row.proposal_episode_id = p2_active_episode_id;
      metric_row.proposal_index_in_episode = p2_active_proposal_index;
      metric_row.proposal_origin = p2_active_proposal_origin;
      metric_row.input_clock_domain = p2_live_metric_receipt.domain();
      metric_row.consumed_steady_ns = p2_live_metric_consumed_ns;
      metric_row.result_ready_steady_ns = fast_livo::p2_receipt::monotonicNs();
      // No checkpoint/prefix/branch strings are guessed from filenames or
      // local event IDs. Until a verified fork supplies them this is unbound.
      auto metric = p2_live_metric_cache;
      if (p2_current_route != route || p2_current_event_id != event_id)
      {
        metric = fast_livo::p2_live_metric::Metric{};
        metric.reason = "consumed_event_context_mismatch";
        metric_row.consumed_steady_ns = 0;
      }
      else if (metric.valid && (p2_live_metric_route != route ||
                               p2_live_metric_event_id != event_id))
      {
        metric = fast_livo::p2_live_metric::Metric{};
        metric.reason = "metric_event_context_mismatch";
      }
      p2_live_metric_receipt.write(metric_row, metric);
    }
  }
}

void LIVMapper::recordP2Proposal(std::uint64_t episode_id,
                                 std::uint64_t proposal_index_in_episode,
                                 bool frozen_forced, double onset_lower_sec,
                                 double onset_point_sec, double onset_upper_sec)
{
  if (!p2_runtime_receipt.enabled()) return;
  p2_active_episode_id = episode_id;
  p2_active_proposal_index = proposal_index_in_episode;
  p2_active_proposal_origin = frozen_forced ? "frozen_forced" : "natural";
  fast_livo::p2_receipt::Row row;
  row.record_type = "proposal";
  row.route = "direct";
  row.event_id = fusion_audit_event;
  row.stage = "detector_trigger_before_arbiter";
  row.modality = p2_current_modality;
  row.sensor_max_input_stamp_ns = p2_current_sensor_max_input_stamp_ns;
  row.input_stamp_semantics = "last_consumed_group_context_not_callback_arrival";
  row.proposal_episode_id = episode_id;
  row.proposal_index_in_episode = proposal_index_in_episode;
  row.proposal_origin = p2_active_proposal_origin;
  row.onset_lower_sec = p2FiniteSeconds(onset_lower_sec);
  row.onset_point_sec = p2FiniteSeconds(onset_point_sec);
  row.onset_upper_sec = p2FiniteSeconds(onset_upper_sec);
  row.buffer_events = map_transaction_buffer.size();
  row.replay_consumed_events = p2_replay_consumed_events;
  row.replay_total_events = p2_replay_total_events;
  row.replay_elapsed_ns = p2_replay_begin_ns == 0 ? 0 :
      (p2_replay_end_ns == 0 ? fast_livo::p2_receipt::monotonicNs() :
       p2_replay_end_ns) - p2_replay_begin_ns;
  row.rearm_count = p2_rearm_count;
  p2_runtime_receipt.write(row);
}

void LIVMapper::shadowAbortCallback(const std_msgs::Bool::ConstPtr &message)
{
  if (!message->data) return;
  ROS_ERROR_STREAM("[ Shadow ] Candidate process received abort request");
  publishTransactionStatus("candidate_terminated", "supervisor_abort");
  ros::shutdown();
}

void LIVMapper::sourceRevisionCutRequestCallback(
    const std_msgs::UInt64::ConstPtr &message)
{
  if (!source_revision_cut_enabled || source_revision_cut_attempted ||
      message->data == 0) return;
  if (source_revision_cut_pending)
  {
    if (message->data != source_revision_cut_switch_stamp_ns)
      ROS_ERROR("[ SourceRevisionCut ] conflicting public switch request ignored");
    return;
  }
  source_revision_cut_switch_stamp_ns = message->data;
  source_revision_cut_request_wall_sec = ros::WallTime::now().toSec();
  source_revision_cut_pending = true;
  ROS_WARN_STREAM("[ SourceRevisionCut ] requested at public switch ns="
                  << source_revision_cut_switch_stamp_ns);
}

void LIVMapper::recoveryProposalCorroborationCallback(
    const std_msgs::String::ConstPtr &message)
{
  const auto evidence =
      fast_livo::recovery_scheduling::parseProposalCorroboration(
          message->data);
  if (!fast_livo::recovery_scheduling::supportsMapRecovery(
          evidence, recovery_proposal_arbiter_config.minimum_raw_residual,
          recovery_proposal_arbiter_config
              .confirmed_lidar_fallback_maximum_innovation_to_lidar_ratio))
    return;
  {
    std::lock_guard<std::mutex> lock(
        recovery_proposal_corroboration_mutex);
    if (!latest_supported_corroboration.complete ||
        evidence.detection_sec >=
            latest_supported_corroboration.detection_sec)
      latest_supported_corroboration = evidence;
    if (pending_external_recovery_proposal.valid &&
        fast_livo::recovery_scheduling::corroboratesCandidateProposal(
            evidence,
            pending_external_recovery_proposal.detection_sec,
            pending_external_recovery_proposal.onset_lower_bound_sec,
            pending_external_recovery_proposal.onset_upper_bound_sec,
            recovery_proposal_corroboration_tolerance_sec))
    {
      pending_external_recovery_proposal.corroboration_ready = true;
      pending_external_recovery_proposal.corroboration_detection_sec =
          evidence.detection_sec;
    }
  }
  ROS_INFO_STREAM("[ RecoveryProposalArbiter ] public corroboration detection="
                  << evidence.detection_sec
                  << " onset_interval=["
                  << evidence.onset_lower_bound_sec << ","
                  << evidence.onset_upper_bound_sec << "]"
                  << " raw_residual=" << evidence.raw_residual
                  << " residual_score=" << evidence.residual_score
                  << " match_score=" << evidence.match_score
                  << " innovation_score=" << evidence.innovation_score);
}

void LIVMapper::processPendingExternalRecoveryProposal(
    const double relative_time_sec)
{
  PendingExternalRecoveryProposal pending;
  bool expired = false;
  {
    std::lock_guard<std::mutex> lock(
        recovery_proposal_corroboration_mutex);
    if (!pending_external_recovery_proposal.valid) return;
    if (!pending_external_recovery_proposal.corroboration_ready)
    {
      if (relative_time_sec -
              pending_external_recovery_proposal.detection_sec <=
          recovery_proposal_corroboration_tolerance_sec)
        return;
      pending = pending_external_recovery_proposal;
      pending_external_recovery_proposal =
          PendingExternalRecoveryProposal();
      expired = true;
    }
    else
    {
      pending = pending_external_recovery_proposal;
      pending_external_recovery_proposal =
          PendingExternalRecoveryProposal();
    }
  }

  if (expired)
  {
    if (std::abs(protected_rolling_checkpoint_time_sec -
                 pending.protected_checkpoint_time_sec) <= 1e-9)
      protected_rolling_checkpoint_time_sec = -1.0;
    publishTransactionStatus(
        "recovery_proposal_suppressed",
        pending.proposal_detail +
            ";arbiter_reason=external_corroboration_timeout" +
            ";expensive_replay_started=0");
    ROS_INFO_STREAM(
        "[ RecoveryProposalArbiter ] pending proposal expired detection="
        << pending.detection_sec);
    return;
  }

  const auto arbiter_decision = recovery_proposal_arbiter->decide(
      pending.episode_id,
      pending.proposal_index_in_episode,
      pending.residual_score,
      pending.match_score,
      pending.innovation_score,
      pending.raw_residual,
      true);
  const std::string external_detail =
      pending.proposal_detail +
      ";external_corroboration_late=1" +
      ";external_corroboration_resolved=1" +
      ";external_corroboration_detection_sec=" +
      std::to_string(pending.corroboration_detection_sec);
  publishTransactionStatus(
      "recovery_proposal_observed",
      external_detail + ";arbiter_reason=" + arbiter_decision.reason +
          ";arbiter_execute_recovery=" +
          std::to_string(static_cast<int>(
              arbiter_decision.execute_recovery)) +
          ";arbiter_external_corroborated_admission=" +
          std::to_string(static_cast<int>(
              arbiter_decision.external_corroborated_admission)));
  if (harm_detector_observe_only || !arbiter_decision.execute_recovery)
  {
    if (std::abs(protected_rolling_checkpoint_time_sec -
                 pending.protected_checkpoint_time_sec) <= 1e-9)
      protected_rolling_checkpoint_time_sec = -1.0;
    publishTransactionStatus(
        "recovery_proposal_suppressed",
        external_detail + ";arbiter_reason=" + arbiter_decision.reason +
            ";expensive_replay_started=0");
    return;
  }

  online_detection_pending = true;
  online_point_onset_sec = pending.point_onset_sec;
  online_detected_onset_sec = pending.onset_lower_bound_sec;
  online_detection_time_sec = pending.detection_sec;
  healthy_residual_limit = pending.healthy_residual_limit;
  healthy_match_ratio_limit = pending.healthy_match_ratio_limit;
  healthy_match_baseline_reference =
      pending.healthy_match_baseline_reference;
  healthy_validator_limits_ready =
      pending.healthy_validator_limits_ready;
  protected_rolling_checkpoint_time_sec = -1.0;
  for (const RollingCheckpoint &checkpoint : rolling_checkpoints)
  {
    if (checkpoint.relative_time < online_detected_onset_sec)
      protected_rolling_checkpoint_time_sec = checkpoint.relative_time;
  }
  lidar_map_writes_enabled = false;
  publishTransactionStatus("recovery_proposed", external_detail);
  ROS_WARN_STREAM(
      "[ RecoveryProposalArbiter ] late public corroboration admitted "
      "candidate detection=" << pending.detection_sec
      << " public_detection=" << pending.corroboration_detection_sec
      << " protected_checkpoint="
      << protected_rolling_checkpoint_time_sec);
}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    if (!std::isfinite(mapper_first_lidar_header_sec))
      throw std::runtime_error("missing first lidar header");
    if (std::isfinite(private_fork_sealed_time_origin_sec) &&
        !fast_livo::replay::sensorEpochMatches(
            mapper_first_lidar_header_sec,
            private_fork_sealed_time_origin_sec))
    {
      std::ostringstream detail;
      detail << std::setprecision(17)
             << "private fork lidar epoch mismatch"
             << ";observed_first_lidar_header_sec="
             << mapper_first_lidar_header_sec
             << ";sealed_time_origin_sec="
             << private_fork_sealed_time_origin_sec;
      ROS_ERROR_STREAM("[ ForkEpoch ] " << detail.str());
      throw std::runtime_error(detail.str());
    }
    // Both source and consumer use the first accepted lidar header. Do not
    // substitute last_lio_update_time or the printed epoch: either one can
    // hide a skipped frame by moving every later relative time.
    _first_lidar_time = mapper_first_lidar_header_sec;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    std::ostringstream origin;
    origin << std::fixed << std::setprecision(17)
           << "[ MapperEpoch ] first_lidar_header_sec=" << _first_lidar_time;
    ROS_INFO_STREAM(origin.str());
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping()
{
  switch (LidarMeasures.lio_vio_flg)
  {
    case VIO:
      if (visual_updates_enabled) handleVIO();
      break;
    case LIO:
    case LO:
      if (lidar_updates_enabled) handleLIO();
      break;
  }
}

void LIVMapper::handleVIO() 
{
  const bool persistent_output_allowed = !map_transaction_state ||
      !map_transaction_state->active() || transaction_replay_active;
  euler_cur = RotMtoEuler(_state.rot_end);
  if (persistent_output_allowed)
  {
    fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
              << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
              << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
  }
    
  if (pcl_w_wait_pub == nullptr || pcl_w_wait_pub->empty())
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) 
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  vio_manager->setPersistentMapWritesEnabled(visual_map_writes_enabled);
  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++) 
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(pubImage, vio_manager);

  euler_cur = RotMtoEuler(_state.rot_end);
  if (persistent_output_allowed)
  {
    fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
              << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
              << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
  }
  last_completed_event_relative_time =
      LidarMeasures.last_lio_update_time - _first_lidar_time;
}

void LIVMapper::handleLIO() 
{    
  const bool persistent_output_allowed = !map_transaction_state ||
      !map_transaction_state->active() || transaction_replay_active;
  euler_cur = RotMtoEuler(_state.rot_end);
  if (persistent_output_allowed)
  {
    fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
             << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
             << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;
  }
           
  if (feats_undistort == nullptr || feats_undistort->empty())
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    if (!lidar_map_writes_enabled)
    {
      ROS_WARN_THROTTLE(1.0, "[ Transaction ] Initial LiDAR-map build is quarantined");
      return;
    }
    voxelmap_manager->BuildVoxelMap();
    lidar_map_inited = true;
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;
  if (p2_live_metric_receipt.enabled() && p2_current_fusion_pending)
  {
    const VD(DIM_STATE) diagnostic_delta = _state - state_propagat;
    p2_live_metric_cache = fast_livo::p2_live_metric::computeObservedPosePriorWhitenedUpdate(
        diagnostic_delta.block<6, 1>(0, 0), state_propagat.cov.block<6, 6>(0, 0),
        voxelmap_manager->effct_feat_num_);
    p2_live_metric_route = p2_current_route;
    p2_live_metric_event_id = p2_current_event_id;
  }
  {
    const VD(DIM_STATE) update_delta = _state - state_propagat;
    const Eigen::Matrix<double, 6, 1> pose_update =
        update_delta.block<6, 1>(0, 0);
    Eigen::Matrix<double, 6, 6> prior_pose_covariance =
        state_propagat.cov.block<6, 6>(0, 0);
    prior_pose_covariance.diagonal().array() += 1e-12;
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(
        prior_pose_covariance);
    if (decomposition.info() == Eigen::Success)
    {
      const Eigen::Matrix<double, 6, 1> whitened =
          decomposition.solve(pose_update);
      const double squared = pose_update.dot(whitened);
      detector_normalized_innovation =
          std::isfinite(squared) && squared >= 0.0 ? std::sqrt(squared) :
          pose_update.norm();
    }
    else
    {
      detector_normalized_innovation = pose_update.norm();
    }
  }

  if (!transaction_replay_active && map_transaction_state &&
      map_transaction_state->phase() == fast_livo::transactional_mapping::Phase::MONITORING &&
      std::isfinite(voxelmap_manager->raw_mean_abs_residual_) &&
      std::isfinite(voxelmap_manager->raw_match_ratio_))
  {
    healthy_residual_history.push_back(voxelmap_manager->raw_mean_abs_residual_);
    healthy_match_history.push_back(voxelmap_manager->raw_match_ratio_);
    while (healthy_residual_history.size() > healthy_baseline_window)
      healthy_residual_history.pop_front();
    while (healthy_match_history.size() > healthy_baseline_window)
      healthy_match_history.pop_front();
  }

  if (online_harm_detector_enabled && !transaction_replay_active)
  {
    const auto transaction_phase = map_transaction_state->phase();
    const double relative_time = LidarMeasures.last_lio_update_time - _first_lidar_time;
    if (transaction_phase == fast_livo::transactional_mapping::Phase::MONITORING)
    {
      auto harm_decision = harm_detector->observe(
          relative_time, voxelmap_manager->raw_mean_abs_residual_,
          voxelmap_manager->raw_match_ratio_, detector_normalized_innovation,
          !frozen_detection_event.enabled);
      if (frozen_detection_event.enabled && !frozen_detection_event.fired &&
          !frozen_detection_event.missed)
      {
        constexpr double kFrozenFrameToleranceSec = 0.002;
        const double frame_delta = relative_time -
            frozen_detection_event.detection_sec;
        if (std::abs(frame_delta) <= kFrozenFrameToleranceSec)
        {
          if (!harm_decision.valid_observation ||
              !harm_decision.baseline_ready)
          {
            frozen_detection_event.missed = true;
            ROS_FATAL_STREAM("[ FrozenEvent ] source frame has no valid detector baseline at t="
                             << relative_time);
            ros::shutdown();
            return;
          }
          harm_decision.triggered = true;
          harm_decision.estimated_onset_sec = frozen_detection_event.point_sec;
          harm_decision.onset_lower_bound_sec = frozen_detection_event.lower_sec;
          harm_decision.onset_upper_bound_sec = frozen_detection_event.upper_sec;
          harm_decision.onset_confidence = frozen_detection_event.confidence;
          harm_decision.onset_uncertainty_sec = std::max(
              harm_decision.estimated_onset_sec -
                  harm_decision.onset_lower_bound_sec,
              harm_decision.onset_upper_bound_sec -
                  harm_decision.estimated_onset_sec);
          harm_decision.episode_id = frozen_detection_event.episode_id;
          harm_decision.proposal_index_in_episode =
              frozen_detection_event.proposal_index_in_episode;
          harm_decision.episode_lower_bound_sec =
              frozen_detection_event.lower_sec;
          harm_decision.episode_upper_bound_sec =
              frozen_detection_event.upper_sec;
          frozen_detection_event.fired = true;
          ROS_WARN_STREAM("[ FrozenEvent ] forced sealed source event at t="
                          << std::setprecision(17) << relative_time
                          << " source_manifest_sha256="
                          << frozen_detection_event.source_manifest_sha256);
        }
        else if (frame_delta > kFrozenFrameToleranceSec)
        {
          frozen_detection_event.missed = true;
          ROS_FATAL_STREAM("[ FrozenEvent ] missed sealed source frame at t="
                           << std::setprecision(17)
                           << frozen_detection_event.detection_sec
                           << "; current t=" << relative_time);
          ros::shutdown();
          return;
        }
      }
      if (harm_detector_diagnostics.is_open())
      {
        harm_detector_diagnostics << std::setprecision(15)
            << relative_time << ','
            << voxelmap_manager->raw_mean_abs_residual_ << ','
            << voxelmap_manager->raw_match_ratio_ << ','
            << detector_normalized_innovation << ','
            << static_cast<int>(harm_decision.valid_observation) << ','
            << static_cast<int>(harm_decision.baseline_ready) << ','
            << static_cast<int>(harm_decision.weak_evidence) << ','
            << static_cast<int>(harm_decision.anomalous) << ','
            << static_cast<int>(harm_decision.triggered) << ','
            << harm_decision.residual_score << ','
            << harm_decision.match_score << ','
            << harm_decision.innovation_score << ','
            << harm_decision.fused_score << ','
            << harm_decision.residual_threshold << ','
            << harm_decision.match_ratio_threshold << ','
            << harm_decision.innovation_threshold << '\n';
        harm_detector_diagnostics.flush();
      }
      if (!harm_decision.valid_observation)
      {
        // An empty/underconstrained scan can produce inf residuals or an
        // undefined match ratio.  Do not learn from it and do not advance a
        // verified-healthy rolling anchor, but keep the estimator alive.
        if (rolling_history_enabled) rolling_checkpoint_rotation_safe = false;
        ROS_WARN_STREAM_THROTTLE(
            1.0, "[ HarmDetector ] skipping invalid observation t="
            << relative_time
            << " raw_residual=" << voxelmap_manager->raw_mean_abs_residual_
            << " raw_match_ratio=" << voxelmap_manager->raw_match_ratio_);
      }
      else
      {
      // Rolling delta anchors may advance only while the detector has no
      // precursor evidence.  Because maintainRollingHistory() snapshots the
      // state before the current event is processed, an anchor opened on the
      // first weak event still represents the last verified healthy state.
      if (rolling_history_enabled)
      {
        rolling_checkpoint_rotation_safe =
            !harm_decision.anomalous &&
            harm_decision.weak_evidence_samples == 0 &&
            !harm_decision.precursor_supported;
      }
      if (harm_decision.baseline_ready)
      {
        ROS_INFO_STREAM_THROTTLE(1.0, "[ HarmDetector ] t=" << relative_time
            << " raw_residual=" << voxelmap_manager->raw_mean_abs_residual_
            << " residual_limit=" << harm_decision.residual_threshold
            << " raw_match_ratio=" << voxelmap_manager->raw_match_ratio_
            << " match_limit=" << harm_decision.match_ratio_threshold
            << " normalized_innovation=" << detector_normalized_innovation
            << " innovation_limit=" << harm_decision.innovation_threshold
            << " fused_score=" << harm_decision.fused_score
            << " effective_residual=" << voxelmap_manager->mean_abs_residual_
            << " effective_match_ratio="
            << voxelmap_manager->effective_match_ratio_
            << " anomalous=" << harm_decision.anomalous);
      }
      if (harm_decision.triggered)
      {
        // Count every detector proposal before the guard, corroborator, or
        // episode arbiter can suppress it. A forced frozen event is explicitly
        // labelled; its episode/index never masquerades as a natural trigger.
        recordP2Proposal(
            harm_decision.episode_id,
            harm_decision.proposal_index_in_episode,
            frozen_detection_event.enabled && frozen_detection_event.fired &&
                std::abs(relative_time -
                         frozen_detection_event.detection_sec) <= 0.002,
            harm_decision.onset_lower_bound_sec,
            harm_decision.estimated_onset_sec,
            harm_decision.onset_upper_bound_sec);
        fast_livo::incremental_public_guard::ProbeResult scale_probe;
        bool scale_probe_confirmed = false;
        if (incremental_guard_config.enabled &&
            harm_decision.onset_confidence >=
                incremental_guard_config.minimum_onset_confidence &&
            !incremental_public_guard.active(relative_time))
        {
          // Counterfactual scan-to-map measurement. Both hypotheses query the
          // same current, pre-write voxel map at the IMU-propagated pose; no
          // candidate state or map is committed by this probe. In particular,
          // the injected fault interval is not an input to the decision.
          std::vector<pointWithVar> identity_points;
          std::vector<pointWithVar> corrected_points;
          const auto &source_points = voxelmap_manager->pv_list_;
          constexpr std::size_t kMaximumProbePoints = 256;
          const std::size_t stride = std::max<std::size_t>(
              1, (source_points.size() + kMaximumProbePoints - 1) /
                     kMaximumProbePoints);
          identity_points.reserve(kMaximumProbePoints);
          corrected_points.reserve(kMaximumProbePoints);
          const double inverse_scale =
              1.0 / incremental_guard_config.terminal_range_scale;
          for (std::size_t index = 0; index < source_points.size();
               index += stride)
          {
            const pointWithVar &source = source_points[index];
            if (!source.point_b.allFinite() || !source.var.allFinite())
              continue;
            pointWithVar identity = source;
            identity.point_w = state_propagat.rot_end *
                (extR * identity.point_b + extT) + state_propagat.pos_end;
            pointWithVar corrected = source;
            corrected.point_b *= inverse_scale;
            corrected.point_w = state_propagat.rot_end *
                (extR * corrected.point_b + extT) + state_propagat.pos_end;
            identity_points.push_back(identity);
            corrected_points.push_back(corrected);
          }
          std::vector<PointToPlane> identity_planes;
          std::vector<PointToPlane> corrected_planes;
          voxelmap_manager->BuildResidualListOMP(identity_points,
                                                 identity_planes);
          voxelmap_manager->BuildResidualListOMP(corrected_points,
                                                 corrected_planes);
          scale_probe.identity_matches = identity_planes.size();
          scale_probe.corrected_matches = corrected_planes.size();
          if (!identity_planes.empty() && !corrected_planes.empty())
          {
            double identity_sum = 0.0;
            double corrected_sum = 0.0;
            for (const auto &plane : identity_planes)
              identity_sum += std::abs(plane.dis_to_plane_);
            for (const auto &plane : corrected_planes)
              corrected_sum += std::abs(plane.dis_to_plane_);
            scale_probe.identity_mean_abs_residual =
                identity_sum / identity_planes.size();
            scale_probe.corrected_mean_abs_residual =
                corrected_sum / corrected_planes.size();
            scale_probe.valid = std::isfinite(
                    scale_probe.identity_mean_abs_residual) &&
                std::isfinite(scale_probe.corrected_mean_abs_residual);
          }
          scale_probe_confirmed = incremental_guard_evidence.observe(
              harm_decision.episode_id, scale_probe);
          ROS_INFO_STREAM("[ IncrementalPublicProbe ] t=" << relative_time
              << " episode_id=" << harm_decision.episode_id
              << " identity_matches=" << scale_probe.identity_matches
              << " corrected_matches=" << scale_probe.corrected_matches
              << " identity_residual="
              << scale_probe.identity_mean_abs_residual
              << " corrected_residual="
              << scale_probe.corrected_mean_abs_residual
              << " consecutive_passes=" << incremental_guard_evidence.passes()
              << " confirmed=" << scale_probe_confirmed);
        }
        if (scale_probe_confirmed && incremental_public_guard.arm(
                relative_time, harm_decision.onset_confidence,
                harm_decision.episode_id))
        {
          std::ostringstream guard_detail;
          guard_detail << std::setprecision(17)
              << "episode_id=" << harm_decision.episode_id
              << ";detection_sec=" << relative_time
              << ";onset_lower_bound_sec="
              << harm_decision.onset_lower_bound_sec
              << ";onset_confidence=" << harm_decision.onset_confidence
              << ";guard_end_sec=" << incremental_public_guard.endSec()
              << ";lidar_information_weight="
              << incremental_public_guard.lidarInformationWeight()
              << ";map_writes_enabled=0;oracle_interval_consumed=0"
              << ";identity_probe_residual="
              << scale_probe.identity_mean_abs_residual
              << ";corrected_probe_residual="
              << scale_probe.corrected_mean_abs_residual;
          publishTransactionStatus("incremental_public_guard_started",
                                   guard_detail.str(), relative_time);
          ROS_WARN_STREAM("[ IncrementalPublicGuard ] armed t="
                          << relative_time << " " << guard_detail.str());
        }
        online_detection_pending = true;
        online_point_onset_sec = harm_decision.estimated_onset_sec;
        online_detected_onset_sec = harm_decision.onset_lower_bound_sec;
        online_detection_time_sec = relative_time;
        if (estimated_range_scale_candidate_enabled)
        {
          fast_livo::correction_estimation::OnsetInterval onset;
          onset.lower_bound_sec = harm_decision.onset_lower_bound_sec;
          onset.point_estimate_sec = harm_decision.estimated_onset_sec;
          onset.upper_bound_sec = harm_decision.onset_upper_bound_sec;
          onset.confidence = harm_decision.onset_confidence;
          onset.detection_sec = relative_time;
          if (onset.confidence >=
              estimated_range_scale_minimum_onset_confidence)
          {
            const auto ramp_start_anchor =
                estimated_range_scale_candidate_ramp_start_anchor == "upper"
                    ? fast_livo::correction_estimation::RampStartAnchor::UPPER_BOUND
                    : (estimated_range_scale_candidate_ramp_start_anchor == "point"
                           ? fast_livo::correction_estimation::RampStartAnchor::POINT_ESTIMATE
                           : fast_livo::correction_estimation::RampStartAnchor::LOWER_BOUND);
            estimated_range_scale_candidate_plan =
                fast_livo::correction_estimation::
                    makeBoundedCandidateReplayPlan(
                        onset,
                        estimated_range_scale_candidate_terminal_scale,
                        estimated_range_scale_post_detection_horizon_sec,
                        estimated_range_scale_candidate_ramp_end_offset_sec,
                        ramp_start_anchor);
          }
          else
          {
            estimated_range_scale_candidate_plan =
                fast_livo::correction_estimation::CorrectionPlan();
            estimated_range_scale_candidate_plan.reason =
                "low_onset_confidence";
          }
          std::ostringstream correction_detail;
          correction_detail << std::setprecision(17)
              << "active=" << static_cast<int>(
                     estimated_range_scale_candidate_plan.active)
              << ";reason=" << estimated_range_scale_candidate_plan.reason
              << ";onset_lower_bound_sec="
              << onset.lower_bound_sec
              << ";point_onset_sec=" << onset.point_estimate_sec
              << ";onset_upper_bound_sec=" << onset.upper_bound_sec
              << ";onset_confidence=" << onset.confidence
              << ";detection_sec=" << onset.detection_sec
              << ";ramp_start_sec="
              << estimated_range_scale_candidate_plan.ramp_start_sec
              << ";ramp_start_anchor="
              << estimated_range_scale_candidate_ramp_start_anchor
              << ";ramp_end_sec="
              << estimated_range_scale_candidate_plan.ramp_end_sec
              << ";ramp_end_offset_sec="
              << estimated_range_scale_candidate_ramp_end_offset_sec
              << ";correction_end_sec="
              << estimated_range_scale_candidate_plan.replay_end_sec
              << ";terminal_range_scale="
              << estimated_range_scale_candidate_terminal_scale
              << ";oracle_interval_consumed=0";
          publishTransactionStatus(
              estimated_range_scale_candidate_plan.active ?
                  "estimated_range_scale_candidate_armed" :
                  "estimated_range_scale_candidate_rejected",
              correction_detail.str(), relative_time);
        }
        protected_rolling_checkpoint_time_sec = -1.0;
        for (const RollingCheckpoint &checkpoint : rolling_checkpoints)
        {
          if (checkpoint.relative_time < online_detected_onset_sec)
            protected_rolling_checkpoint_time_sec = checkpoint.relative_time;
        }
        if (protected_rolling_checkpoint_time_sec >= 0.0)
        {
          ROS_WARN_STREAM("[ RollingHistory ] protected checkpoint t="
                          << protected_rolling_checkpoint_time_sec
                          << " before onset lower bound t="
                          << online_detected_onset_sec);
        }
        healthy_residual_limit = harm_decision.residual_median +
            std::max(healthy_residual_mad_multiplier * harm_decision.residual_sigma,
                     healthy_residual_increase_floor);
        healthy_match_ratio_limit = harm_decision.match_ratio_median -
            std::max(healthy_match_mad_multiplier * harm_decision.match_ratio_sigma,
                     healthy_match_drop_floor);
        healthy_match_ratio_limit = std::max(0.0, healthy_match_ratio_limit);
        healthy_match_baseline_reference = harm_decision.match_ratio_median;
        healthy_validator_limits_ready = true;
        // The triggering frame has already constrained the transient state, but
        // must not gain a persistent map side effect before recovery activates.
        lidar_map_writes_enabled = false;
        ROS_WARN_STREAM("[ HarmDetector ] TRIGGER detection=" << online_detection_time_sec
            << " point_onset=" << online_point_onset_sec
            << " onset_lower_bound=" << online_detected_onset_sec
            << " onset_upper_bound=" << harm_decision.onset_upper_bound_sec
            << " onset_confidence=" << harm_decision.onset_confidence
            << " onset_guard=" << harm_decision.onset_uncertainty_sec
            << " anomaly_duration=" << harm_decision.anomaly_duration_sec
            << " change_point_found=" << harm_decision.change_point_found
            << " precursor_supported=" << harm_decision.precursor_supported
            << " change_point_interval=[" << harm_decision.change_point_lower_sec
            << "," << harm_decision.change_point_upper_sec << "]"
            << " weak_evidence_samples=" << harm_decision.weak_evidence_samples
            << " episode_id=" << harm_decision.episode_id
            << " proposal_index_in_episode="
            << harm_decision.proposal_index_in_episode
            << " frozen_event_forced="
            << static_cast<int>(frozen_detection_event.enabled &&
                                frozen_detection_event.fired)
            << " frozen_source_manifest_sha256="
            << (frozen_detection_event.enabled ?
                frozen_detection_event.source_manifest_sha256 : "none")
            << " merged_with_previous_proposal="
            << harm_decision.merged_with_previous_proposal
            << " residual_score=" << harm_decision.residual_score
            << " match_score=" << harm_decision.match_score
            << " innovation_score=" << harm_decision.innovation_score
            << " fused_score=" << harm_decision.fused_score
            << " raw_residual=" << voxelmap_manager->raw_mean_abs_residual_
            << " raw_match_ratio=" << voxelmap_manager->raw_match_ratio_
            << " effective_residual=" << voxelmap_manager->mean_abs_residual_
            << " effective_match_ratio="
            << voxelmap_manager->effective_match_ratio_);
        std::ostringstream proposal;
        proposal << "detection_sec=" << online_detection_time_sec
                 << ";point_onset_sec=" << online_point_onset_sec
                 << ";onset_lower_bound_sec=" << online_detected_onset_sec
                 << ";onset_upper_bound_sec="
                 << harm_decision.onset_upper_bound_sec
                 << ";onset_confidence=" << harm_decision.onset_confidence
                 << ";onset_interval_semantics="
                 << harm_decision.onset_interval_semantics
                 << ";episode_id=" << harm_decision.episode_id
                 << ";proposal_index_in_episode="
                 << harm_decision.proposal_index_in_episode
                 << ";merged_with_previous_proposal="
                 << static_cast<int>(harm_decision.merged_with_previous_proposal)
                 << ";episode_lower_bound_sec="
                 << harm_decision.episode_lower_bound_sec
                 << ";episode_upper_bound_sec="
                 << harm_decision.episode_upper_bound_sec
                 << ";residual_score=" << harm_decision.residual_score
                 << ";raw_residual="
                 << voxelmap_manager->raw_mean_abs_residual_
                 << ";raw_match_ratio="
                 << voxelmap_manager->raw_match_ratio_
                 << ";match_score=" << harm_decision.match_score
                 << ";innovation_score=" << harm_decision.innovation_score
                 << ";fused_score=" << harm_decision.fused_score
                 << ";anomaly_duration_sec="
                 << harm_decision.anomaly_duration_sec
                 << ";observe_only=" << static_cast<int>(harm_detector_observe_only);
        bool external_corroboration = true;
        double corroboration_detection_sec = -1.0;
        if (recovery_proposal_arbiter_config.require_external_corroboration)
        {
          std::lock_guard<std::mutex> lock(
              recovery_proposal_corroboration_mutex);
          if (latest_supported_corroboration.complete)
          {
            corroboration_detection_sec =
                latest_supported_corroboration.detection_sec;
            external_corroboration =
                fast_livo::recovery_scheduling::
                    corroboratesCandidateProposal(
                        latest_supported_corroboration,
                        online_detection_time_sec,
                        online_detected_onset_sec,
                        harm_decision.onset_upper_bound_sec,
                        recovery_proposal_corroboration_tolerance_sec);
          }
          else
          {
            external_corroboration = false;
          }
        }
        proposal << ";external_corroboration="
                 << static_cast<int>(external_corroboration)
                 << ";external_corroboration_detection_sec="
                 << corroboration_detection_sec;
        const auto arbiter_decision = recovery_proposal_arbiter->decide(
            harm_decision.episode_id,
            harm_decision.proposal_index_in_episode,
            harm_decision.residual_score,
            harm_decision.match_score,
            harm_decision.innovation_score,
            voxelmap_manager->raw_mean_abs_residual_,
            external_corroboration);
        const bool awaiting_external_corroboration =
            recovery_proposal_arbiter_config.require_external_corroboration &&
            !external_corroboration &&
            arbiter_decision.reason == "external_corroboration_pending";
        if (awaiting_external_corroboration)
        {
          std::lock_guard<std::mutex> lock(
              recovery_proposal_corroboration_mutex);
          pending_external_recovery_proposal.valid = true;
          pending_external_recovery_proposal.corroboration_ready = false;
          pending_external_recovery_proposal.episode_id =
              harm_decision.episode_id;
          pending_external_recovery_proposal.proposal_index_in_episode =
              harm_decision.proposal_index_in_episode;
          pending_external_recovery_proposal.detection_sec =
              online_detection_time_sec;
          pending_external_recovery_proposal.point_onset_sec =
              online_point_onset_sec;
          pending_external_recovery_proposal.onset_lower_bound_sec =
              online_detected_onset_sec;
          pending_external_recovery_proposal.onset_upper_bound_sec =
              harm_decision.onset_upper_bound_sec;
          pending_external_recovery_proposal.residual_score =
              harm_decision.residual_score;
          pending_external_recovery_proposal.match_score =
              harm_decision.match_score;
          pending_external_recovery_proposal.innovation_score =
              harm_decision.innovation_score;
          pending_external_recovery_proposal.raw_residual =
              voxelmap_manager->raw_mean_abs_residual_;
          pending_external_recovery_proposal.healthy_residual_limit =
              healthy_residual_limit;
          pending_external_recovery_proposal.healthy_match_ratio_limit =
              healthy_match_ratio_limit;
          pending_external_recovery_proposal
              .healthy_match_baseline_reference =
              healthy_match_baseline_reference;
          pending_external_recovery_proposal
              .healthy_validator_limits_ready =
              healthy_validator_limits_ready;
          pending_external_recovery_proposal
              .protected_checkpoint_time_sec =
              protected_rolling_checkpoint_time_sec;
          pending_external_recovery_proposal.corroboration_detection_sec =
              -1.0;
          pending_external_recovery_proposal.proposal_detail = proposal.str();
        }
        if (recovery_proposal_arbiter_config.enabled)
        {
          publishTransactionStatus(
              "recovery_proposal_observed",
              proposal.str() + ";arbiter_reason=" + arbiter_decision.reason +
                  ";arbiter_execute_recovery=" +
                  std::to_string(static_cast<int>(
                      arbiter_decision.execute_recovery)) +
                  ";arbiter_attempts_in_episode=" +
                  std::to_string(arbiter_decision.attempts_in_episode) +
                  ";arbiter_escalation_retry=" +
                  std::to_string(static_cast<int>(
                      arbiter_decision.escalation_retry)) +
                  ";arbiter_rejected_history_escalation=" +
                  std::to_string(static_cast<int>(
                      arbiter_decision.rejected_history_escalation)) +
                  ";arbiter_continue_observing_episode=" +
                  std::to_string(static_cast<int>(
                      arbiter_decision.continue_observing_episode)) +
                  ";arbiter_confirmed_lidar_fallback=" +
                  std::to_string(static_cast<int>(
                      arbiter_decision.confirmed_lidar_fallback)) +
                  ";arbiter_confirmed_lidar_evidence_observed=" +
                  std::to_string(static_cast<int>(
                      arbiter_decision.confirmed_lidar_evidence_observed)) +
                  ";arbiter_external_corroboration=" +
                  std::to_string(static_cast<int>(
                      arbiter_decision.external_corroboration)) +
                  ";arbiter_external_corroborated_admission=" +
                  std::to_string(static_cast<int>(
                      arbiter_decision.external_corroborated_admission)));
        }
        if (arbiter_decision.execute_recovery)
          publishTransactionStatus("recovery_proposed", proposal.str());
        else
          publishTransactionStatus(
              "recovery_proposal_suppressed",
              proposal.str() + ";arbiter_reason=" + arbiter_decision.reason +
                  ";expensive_replay_started=0");
        if (harm_detector_observe_only || !arbiter_decision.execute_recovery)
        {
          // Observe-only and episode-budget suppression both retain the full
          // proposal evidence without starting an expensive replay.
          online_detection_pending = false;
          online_point_onset_sec = -1.0;
          online_detected_onset_sec = -1.0;
          online_detection_time_sec = -1.0;
          if (!awaiting_external_corroboration)
            protected_rolling_checkpoint_time_sec = -1.0;
          lidar_map_writes_enabled = true;
          if (harm_detector_observe_only ||
              arbiter_decision.continue_observing_episode)
            harm_detector->resetForMonitoring();
          else
            harm_detector->suppressUntilHealthyTransition();
        }
      }
      }
    }
  }

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // The append-only journal is an experiment/audit output and must not depend
  // on the legacy evo text export switch.  Shadow runs intentionally disable
  // the latter to avoid two processes racing on Log/result/<seq>.txt.
  if (pose_output_en || trajectory_journal.is_open())
  {
    Eigen::Quaterniond q(_state.rot_end);
    if (pose_output_en)
    {
      trajectory_poses[LidarMeasures.last_lio_update_time] = {
          _state.pos_end[0], _state.pos_end[1], _state.pos_end[2],
          q.x(), q.y(), q.z(), q.w()};
    }
    if (trajectory_journal.is_open())
    {
      std::ostringstream stamp_key;
      stamp_key << std::setprecision(17) << LidarMeasures.last_lio_update_time;
      const bool already_recorded =
          production_replay_trajectory_stamps.count(stamp_key.str()) != 0;
      const int trajectory_flag = fast_livo::trajectory_journal::selectFlag(
          incumbent_rebuild_trajectory_active, transaction_replay_active,
          map_transaction_state && map_transaction_state->active(),
          already_recorded);
      if (trajectory_flag == 1)
        production_replay_trajectory_stamps.insert(stamp_key.str());
      trajectory_journal << std::setprecision(15) << LidarMeasures.last_lio_update_time << ' '
                         << _state.pos_end[0] << ' ' << _state.pos_end[1] << ' '
                         << _state.pos_end[2] << ' ' << q.x() << ' ' << q.y() << ' '
                         << q.z() << ' ' << q.w() << ' '
                         << trajectory_flag
                         << '\n';
      trajectory_journal.flush();
    }
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) 
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  if (lidar_map_writes_enabled)
  {
    if (transaction_replay_active &&
        (voxelmap_manager->selective_residual_gate_active_ ||
         selective_map_write_gate_active))
    {
      std::vector<pointWithVar> selective_map_points;
      if (selective_map_write_gate_active &&
          !voxelmap_manager->selective_residual_gate_active_)
      {
        std::vector<double> residuals;
        residuals.reserve(voxelmap_manager->ptpl_list_.size());
        for (const PointToPlane &candidate : voxelmap_manager->ptpl_list_)
          residuals.push_back(std::abs(candidate.dis_to_plane_));
        const auto gate = fast_livo::selective_gating::select(
            residuals, selective_lidar_gate_config);
        selective_map_points.reserve(gate.retained_count);
        for (std::size_t index = 0;
             index < voxelmap_manager->ptpl_list_.size(); ++index)
        {
          const PointToPlane &inlier = voxelmap_manager->ptpl_list_[index];
          if (gate.keep[index] &&
              inlier.source_index_ < voxelmap_manager->pv_list_.size())
            selective_map_points.push_back(
                voxelmap_manager->pv_list_[inlier.source_index_]);
        }
        std::cout << "[ LIO ] Selective map-only gate threshold: "
                  << gate.threshold_m << " m, retained "
                  << gate.retained_count << "/" << gate.keep.size()
                  << " correspondences" << std::endl;
      }
      else
      {
        selective_map_points.reserve(voxelmap_manager->ptpl_list_.size());
        for (const PointToPlane &inlier : voxelmap_manager->ptpl_list_)
        {
          if (inlier.source_index_ < voxelmap_manager->pv_list_.size())
            selective_map_points.push_back(
                voxelmap_manager->pv_list_[inlier.source_index_]);
        }
      }
      voxelmap_manager->UpdateVoxelMap(selective_map_points);
      std::cout << "[ LIO ] Selective replay map write: "
                << selective_map_points.size() << "/"
                << voxelmap_manager->pv_list_.size() << " points" << std::endl;
    }
    else
    {
      voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
    }
    std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  }
  else
  {
    std::cout << "[ LIO ] Persistent voxel-map write quarantined" << std::endl;
  }
  _pv_list = voxelmap_manager->pv_list_;
  
  double t4 = omp_get_wtime();

  if(lidar_map_writes_enabled && voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  if (persistent_output_allowed)
  {
    fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
              << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
              << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
  }
  last_completed_event_relative_time =
      LidarMeasures.last_lio_update_time - _first_lidar_time;
}

bool LIVMapper::writeSourceRevisionFreezeReceipt(
    const std::string &raw_map_path, const std::size_t map_point_count)
{
  if (!sourceRevisionReceiptEnabled()) return true;
  try
  {
    if (map_point_count == 0 || path.poses.empty() ||
        !std::isfinite(_first_lidar_time) ||
        !std::isfinite(last_completed_event_relative_time) ||
        last_completed_event_relative_time < 0.0 ||
        path.header.frame_id != "camera_init" ||
        path.header.stamp != path.poses.back().header.stamp)
      throw std::runtime_error("incomplete source path/map freeze state");
    ros::Time freeze_time;
    freeze_time.fromSec(_first_lidar_time + last_completed_event_relative_time);
    const std::uint64_t watermark_ns = freeze_time.toNSec();
    const std::uint64_t path_final_ns = path.header.stamp.toNSec();
    const std::uint64_t gap_ns = watermark_ns > path_final_ns ?
        watermark_ns - path_final_ns : path_final_ns - watermark_ns;
    if (gap_ns > 200000000ULL)
      throw std::runtime_error("final path is not at map freeze watermark");
    for (std::size_t index = 0; index < path.poses.size(); ++index)
    {
      const auto &pose = path.poses[index];
      if (pose.header.frame_id != "camera_init" ||
          (index > 0 &&
           path.poses[index - 1].header.stamp >= pose.header.stamp))
        throw std::runtime_error("source path has mixed frame/order");
    }
    if (!syncFilePath(raw_map_path))
      throw std::runtime_error("full PCD could not be made durable");
    const std::string source_path = joinPath(
        pcd_output_directory, "source_frozen_path.txt");
    const std::string receipt_path = joinPath(
        pcd_output_directory, "source_revision.freeze");
    if (std::filesystem::exists(source_path) ||
        std::filesystem::exists(receipt_path))
      throw std::runtime_error("source revision already exists");
    const std::string path_tmp = source_path + ".tmp";
    const std::string receipt_tmp = receipt_path + ".tmp";
    {
      std::ofstream output(path_tmp, std::ios::out | std::ios::trunc);
      output << std::setprecision(17);
      for (const auto &pose : path.poses)
        output << pose.header.stamp.toNSec() << ' '
               << pose.pose.position.x << ' ' << pose.pose.position.y << ' '
               << pose.pose.position.z << ' ' << pose.pose.orientation.x << ' '
               << pose.pose.orientation.y << ' ' << pose.pose.orientation.z << ' '
               << pose.pose.orientation.w << '\n';
      output.close();
      if (!output || !syncFilePath(path_tmp))
        throw std::runtime_error("source path write/fsync failed");
    }
    const std::string revision_id = "mapper_freeze_" +
        std::to_string(watermark_ns) + "_" +
        std::to_string(path.poses.size()) + "_" +
        std::to_string(map_point_count);
    {
      std::ofstream receipt(receipt_tmp, std::ios::out | std::ios::trunc);
      receipt << "schema=1\n"
              << "scope=mapper_source_freeze_not_public_commit\n"
              << "source_frame=camera_init\n"
              << "revision_id=" << revision_id << '\n'
              << "sensor_watermark_ns=" << watermark_ns << '\n'
              << "path_final_ns=" << path_final_ns << '\n'
              << "path_pose_count=" << path.poses.size() << '\n'
              << "map_point_count=" << map_point_count << '\n'
              << "raw_map=" << raw_map_path << '\n'
              << "source_path=" << source_path << '\n';
      receipt.close();
      if (!receipt || !syncFilePath(receipt_tmp))
        throw std::runtime_error("source freeze receipt write/fsync failed");
    }
    std::filesystem::rename(path_tmp, source_path);
    std::filesystem::rename(receipt_tmp, receipt_path);
    if (!syncDirectoryPath(pcd_output_directory))
      throw std::runtime_error("source freeze directory fsync failed");
    ROS_INFO_STREAM("[ SourceRevision ] frozen source revision " << revision_id);
    return true;
  }
  catch (const std::exception &error)
  {
    ROS_ERROR_STREAM("[ SourceRevision ] freeze receipt failed: " << error.what());
    return false;
  }
}

void LIVMapper::maybeWriteSourceRevisionCut()
{
  if (!source_revision_cut_pending || source_revision_cut_attempted ||
      path.poses.empty() || path.header.stamp.toNSec() <
          source_revision_cut_switch_stamp_ns)
    return;
  // spinOnce and the mapping loop share this thread.  The preceding fusion
  // event has completed, while the next event has not yet begun; path and the
  // accumulated full PCD therefore describe one immutable source boundary.
  source_revision_cut_attempted = true;
  source_revision_cut_pending = false;
  const std::string directory = joinPath(
      pcd_output_directory,
      "cut_revision_" + std::to_string(source_revision_cut_switch_stamp_ns));
  try
  {
    if (!pcd_save_en || bounded_voxel_export_en || pcd_save_interval >= 0 ||
        pcd_save_type != 0 ||
        (map_transaction_state && map_transaction_state->active()) ||
        transaction_replay_active || !std::isfinite(_first_lidar_time) ||
        !std::isfinite(last_completed_event_relative_time) ||
        path.header.frame_id != "camera_init" ||
        path.header.stamp != path.poses.back().header.stamp)
      throw std::runtime_error("source cut prerequisites are not satisfied");
    const std::size_t map_points = img_en ? pcl_wait_save->size() :
                                            pcl_wait_save_intensity->size();
    if (map_points == 0)
      throw std::runtime_error("source cut has no complete-map points");
    ros::Time event_time;
    event_time.fromSec(_first_lidar_time + last_completed_event_relative_time);
    const std::uint64_t watermark_ns = event_time.toNSec();
    const std::uint64_t path_final_ns = path.header.stamp.toNSec();
    const std::uint64_t gap_ns = watermark_ns > path_final_ns ?
        watermark_ns - path_final_ns : path_final_ns - watermark_ns;
    if (gap_ns > 1000ULL ||
        path_final_ns < source_revision_cut_switch_stamp_ns)
      throw std::runtime_error("source path/map have no common cut watermark");
    if (source_revision_barrier_enabled &&
        (watermark_ns != source_revision_cut_switch_stamp_ns ||
         path_final_ns != source_revision_cut_switch_stamp_ns))
      throw std::runtime_error("pre-public source cut is not at the held odometry stamp");
    for (std::size_t index = 0; index < path.poses.size(); ++index)
      if (path.poses[index].header.frame_id != "camera_init" ||
          (index > 0 && path.poses[index - 1].header.stamp >=
                          path.poses[index].header.stamp))
        throw std::runtime_error("source cut path has mixed frame/order");
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("source cut directory already exists");
    const std::string raw_map_tmp = joinPath(directory, "full_map.pcd.tmp");
    const std::string raw_map_path = joinPath(directory, "full_map.pcd");
    const std::string source_path_tmp = joinPath(directory, "source_path.txt.tmp");
    const std::string source_path = joinPath(directory, "source_path.txt");
    const std::string receipt_tmp = joinPath(directory, "source_revision.cut.tmp");
    const std::string receipt_path = joinPath(directory, "source_revision.cut");
    pcl::PCDWriter writer;
    const bool map_ok = img_en ?
        writer.writeBinary(raw_map_tmp, *pcl_wait_save) == 0 :
        writer.writeBinary(raw_map_tmp, *pcl_wait_save_intensity) == 0;
    if (!map_ok || !syncFilePath(raw_map_tmp))
      throw std::runtime_error("source cut map write/fsync failed");
    {
      std::ofstream output(source_path_tmp, std::ios::out | std::ios::trunc);
      output << std::setprecision(17);
      for (const auto &pose : path.poses)
        output << pose.header.stamp.toNSec() << ' '
               << pose.pose.position.x << ' ' << pose.pose.position.y << ' '
               << pose.pose.position.z << ' ' << pose.pose.orientation.x << ' '
               << pose.pose.orientation.y << ' ' << pose.pose.orientation.z << ' '
               << pose.pose.orientation.w << '\n';
      output.close();
      if (!output || !syncFilePath(source_path_tmp))
        throw std::runtime_error("source cut path write/fsync failed");
    }
    const std::string revision_id = "mapper_cut_" +
        std::to_string(source_revision_cut_switch_stamp_ns) + "_" +
        std::to_string(watermark_ns) + "_" +
        std::to_string(path.poses.size()) + "_" +
        std::to_string(map_points);
    const std::string raw_map_sha256 = source_revision_payload_hash_enabled ?
        fast_livo::source_revision::sha256File(raw_map_tmp) : std::string();
    const std::string source_path_sha256 = source_revision_payload_hash_enabled ?
        fast_livo::source_revision::sha256File(source_path_tmp) : std::string();
    {
      std::ofstream output(receipt_tmp, std::ios::out | std::ios::trunc);
      output << std::setprecision(17)
             << "schema=" << (source_revision_payload_hash_enabled ? 2 : 1)
             << '\n'
             << "scope=" << (source_revision_barrier_enabled ?
                 "mapper_prepublic_cut_for_mux_barrier" :
                 "mapper_cut_snapshot_not_public_commit") << '\n'
             << "source_frame=camera_init\n"
             << "revision_id=" << revision_id << '\n'
             << "requested_public_switch_ns="
             << source_revision_cut_switch_stamp_ns << '\n'
             << "sensor_watermark_ns=" << watermark_ns << '\n'
             << "path_final_ns=" << path_final_ns << '\n'
             << "path_pose_count=" << path.poses.size() << '\n'
             << "map_point_count=" << map_points << '\n'
             << "request_received_wall_sec="
             << source_revision_cut_request_wall_sec << '\n'
             << "payload_written_wall_sec=" << ros::WallTime::now().toSec()
             << '\n'
             << "raw_map=" << raw_map_path << '\n'
             << "source_path=" << source_path << '\n';
      if (source_revision_payload_hash_enabled)
        output << "raw_map_sha256=" << raw_map_sha256 << '\n'
               << "source_path_sha256=" << source_path_sha256 << '\n';
      output.close();
      if (!output || !syncFilePath(receipt_tmp))
        throw std::runtime_error("source cut receipt write/fsync failed");
    }
    std::filesystem::rename(raw_map_tmp, raw_map_path);
    std::filesystem::rename(source_path_tmp, source_path);
    std::filesystem::rename(receipt_tmp, receipt_path);
    if (!syncDirectoryPath(directory))
      throw std::runtime_error("source cut directory fsync failed");
    const std::string completed_path = joinPath(directory,
                                                "cut_revision.completed");
    const std::string completed_tmp = completed_path + ".tmp";
    {
      std::ofstream output(completed_tmp, std::ios::out | std::ios::trunc);
      output << std::setprecision(17)
             << "revision_id=" << revision_id << '\n'
             << "marker_written_wall_sec=" << ros::WallTime::now().toSec()
             << '\n';
      output.close();
      if (!output || !syncFilePath(completed_tmp))
        throw std::runtime_error("source cut completion marker failed");
    }
    std::filesystem::rename(completed_tmp, completed_path);
    if (!syncDirectoryPath(directory))
      throw std::runtime_error("source cut completion directory fsync failed");
    if (source_revision_barrier_enabled)
    {
      const std::string receipt_sha256 = source_revision_payload_hash_enabled ?
          fast_livo::source_revision::sha256File(receipt_path) : std::string();
      const std::string completed_sha256 = source_revision_payload_hash_enabled ?
          fast_livo::source_revision::sha256File(completed_path) : std::string();
      std_msgs::String ready;
      ready.data = "schema=" +
          std::to_string(source_revision_payload_hash_enabled ? 2 : 1) +
          ";status=ready;requested_public_switch_ns=" +
          std::to_string(source_revision_cut_switch_stamp_ns) +
          ";revision_id=" + revision_id +
          ";receipt=" + receipt_path +
          ";completed=" + completed_path;
      if (source_revision_payload_hash_enabled)
        ready.data += ";raw_map_sha256=" + raw_map_sha256 +
            ";source_path_sha256=" + source_path_sha256 +
            ";receipt_sha256=" + receipt_sha256 +
            ";completed_sha256=" + completed_sha256;
      source_revision_cut_ready_pub.publish(ready);
    }
    ROS_WARN_STREAM("[ SourceRevisionCut ] durable cut revision=" << revision_id
                    << " switch_to_watermark_ns="
                    << watermark_ns - source_revision_cut_switch_stamp_ns);
  }
  catch (const std::exception &error)
  {
    ROS_ERROR_STREAM("[ SourceRevisionCut ] failed: " << error.what());
    const std::string failed_path = joinPath(
        pcd_output_directory,
        "cut_revision_" +
            std::to_string(source_revision_cut_switch_stamp_ns) + ".failed");
    std::ofstream failed(failed_path, std::ios::out | std::ios::trunc);
    failed << error.what() << '\n';
    failed.close();
    if (source_revision_barrier_enabled)
    {
      std_msgs::String failure;
      failure.data = "schema=" +
          std::to_string(source_revision_payload_hash_enabled ? 2 : 1) +
          ";status=failed;requested_public_switch_ns=" +
          std::to_string(source_revision_cut_switch_stamp_ns);
      source_revision_cut_ready_pub.publish(failure);
    }
  }
}

void LIVMapper::savePCD() 
{
  saveTrajectoryPoses();
  if (pcd_save_en && bounded_voxel_export_en)
  {
    const std::string raw_points_dir =
        joinPath(pcd_output_directory, "all_raw_points.pcd");
    const std::string downsampled_points_dir =
        joinPath(pcd_output_directory, "all_downsampled_points.pcd");
    PointCloudXYZI compact_map;
    const auto statistics = voxelmap_manager->ExportRepresentativeCloud(
        compact_map, bounded_voxel_export_max_points,
        bounded_voxel_export_samples_per_plane);
    pcl::PCDWriter pcd_writer;
    const bool nonempty = !compact_map.empty();
    const bool raw_ok = nonempty &&
        pcd_writer.writeBinary(raw_points_dir, compact_map) == 0;
    const bool downsampled_ok = raw_ok &&
        pcd_writer.writeBinary(downsampled_points_dir, compact_map) == 0;
    const bool write_ok = raw_ok && downsampled_ok &&
        writeSourceRevisionFreezeReceipt(raw_points_dir,
                                         compact_map.points.size());
    const std::string final_marker_path = joinPath(
        pcd_output_directory,
        write_ok ? "pcd_save.completed" : "pcd_save.failed");
    std::ofstream marker(final_marker_path,
                         std::ios::out | std::ios::trunc);
    marker << "export_mode=bounded_voxel_surfel\n";
    marker << "raw_map=" << raw_points_dir << "\n";
    marker << "downsampled_map=" << downsampled_points_dir << "\n";
    marker << "root_voxels=" << statistics.root_voxels << "\n";
    marker << "terminal_nodes=" << statistics.terminal_nodes << "\n";
    marker << "plane_nodes=" << statistics.plane_nodes << "\n";
    marker << "fallback_nodes=" << statistics.fallback_nodes << "\n";
    marker << "emitted_points=" << statistics.emitted_points << "\n";
    marker << "maximum_points=" << statistics.maximum_points << "\n";
    marker << "truncated=" << (statistics.truncated ? 1 : 0) << "\n";
    marker.flush();
    marker.close();
    if (sourceRevisionReceiptEnabled() &&
        (!syncFilePath(final_marker_path) ||
         !syncDirectoryPath(pcd_output_directory)))
      std::_Exit(EXIT_FAILURE);
    ROS_INFO_STREAM("[ MapExport ] bounded voxel export roots="
                    << statistics.root_voxels
                    << " terminals=" << statistics.terminal_nodes
                    << " points=" << statistics.emitted_points
                    << " truncated=" << statistics.truncated
                    << " marker=" << final_marker_path);
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    std::_Exit(write_ok ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    const std::string raw_points_dir = joinPath(pcd_output_directory, "all_raw_points.pcd");
    const std::string downsampled_points_dir =
        joinPath(pcd_output_directory, "all_downsampled_points.pcd");
    pcl::PCDWriter pcd_writer;

    bool map_write_ok = true;
    const auto complete_map_save_and_exit =
        [&](const bool write_ok, const bool has_downsampled) {
          const bool receipt_ok = write_ok &&
              writeSourceRevisionFreezeReceipt(
                  raw_points_dir, img_en ? pcl_wait_save->points.size() :
                                        pcl_wait_save_intensity->points.size());
          const bool final_ok = write_ok && receipt_ok;
          const std::string marker_path =
              joinPath(pcd_output_directory,
                       final_ok ? "pcd_save.completed" : "pcd_save.failed");
          std::ofstream marker(marker_path, std::ios::out | std::ios::trunc);
          marker << "raw_map=" << raw_points_dir << "\n";
          if (has_downsampled)
            marker << "downsampled_map=" << downsampled_points_dir << "\n";
          marker.flush();
          marker.close();
          if (sourceRevisionReceiptEnabled() &&
              (!syncFilePath(marker_path) ||
               !syncDirectoryPath(pcd_output_directory)))
            std::_Exit(EXIT_FAILURE);
          std::cout.flush();
          std::cerr.flush();
          std::fflush(nullptr);
          std::_Exit(final_ok ? EXIT_SUCCESS : EXIT_FAILURE);
        };
    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
  
      map_write_ok = pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save) == 0;
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      map_write_ok = pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud) == 0 && map_write_ok;
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
      // Do not leave this scope: destruction of the temporary PCL filter/cloud
      // is the post-save SIGSEGV reproduced on both isolated mapper branches.
      complete_map_save_and_exit(map_write_ok, true);
    }
    else
    {      
      map_write_ok = pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity) == 0;
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
      complete_map_save_and_exit(map_write_ok, false);
    }
  }
}

void LIVMapper::exportPrivateCandidateMapReceipt(const double replay_boundary_time)
{
  if (!private_candidate_map_receipt_enabled) return;
  try
  {
    if (!transaction_replay_active || !frozen_detection_event.enabled ||
        !frozen_detection_event.fired || private_candidate_handoff_generation == 0 ||
        !fast_livo::source_revision::validSha256(private_candidate_checkpoint_sha256) ||
        !fast_livo::source_revision::validSha256(frozen_detection_event.source_manifest_sha256))
      throw std::runtime_error("private receipt requires a bound frozen proposal and imported checkpoint");
    if (!std::isfinite(last_completed_event_relative_time) ||
        std::abs(last_completed_event_relative_time - replay_boundary_time) > 2e-6)
      throw std::runtime_error("replay boundary is not the completed mapper state");
    std::ostringstream name;
    name << "generation_" << private_candidate_handoff_generation
         << "_episode_" << frozen_detection_event.episode_id
         << "_proposal_" << frozen_detection_event.proposal_index_in_episode;
    const auto directory = std::filesystem::path(private_candidate_map_receipt_directory) / name.str();
    std::filesystem::create_directories(directory.parent_path());
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("private snapshot directory already exists");
    PointCloudXYZI cloud;
    const auto stats = voxelmap_manager->ExportRepresentativeCloud(
        cloud, rejected_candidate_map_export_max_points,
        rejected_candidate_map_export_samples_per_plane);
    const auto map_path = directory / "candidate_map.pcd";
    if (cloud.empty() || pcl::PCDWriter().writeBinary(map_path.string(), cloud) != 0)
      throw std::runtime_error("private candidate map write failed");
    auto fields = fast_livo::private_map_receipt::runtimeHashes();
    const auto number = [](double value) {
      std::ostringstream out;
      out << std::setprecision(17) << value;
      return out.str();
    };
    fields["source_manifest_sha256"] = frozen_detection_event.source_manifest_sha256;
    fields["checkpoint_sha256"] = private_candidate_checkpoint_sha256;
    fields["generation"] = std::to_string(private_candidate_handoff_generation);
    fields["episode_id"] = std::to_string(frozen_detection_event.episode_id);
    fields["proposal_index_in_episode"] = std::to_string(frozen_detection_event.proposal_index_in_episode);
    fields["checkpoint_completed_relative_sec"] = number(map_transaction_checkpoint.last_completed_event_relative_time);
    fields["state_watermark_relative_sec"] = number(last_completed_event_relative_time);
    fields["replay_boundary_relative_sec"] = number(replay_boundary_time);
    fields["candidate_terminal_scale"] = number(estimated_range_scale_candidate_terminal_scale);
    fields["reference_frame"] = "camera_init";
    fields["export_policy"] = "bounded_voxel_surfel_v1";
    fields["maximum_points"] = std::to_string(rejected_candidate_map_export_max_points);
    fields["samples_per_plane"] = std::to_string(rejected_candidate_map_export_samples_per_plane);
    fields["emitted_points"] = std::to_string(stats.emitted_points);
    fields["truncated"] = stats.truncated ? "1" : "0";
    if (!syncFilePath(map_path.string()))
      throw std::runtime_error("private map sync failed");
    const auto receipt = directory / "map.receipt";
    fast_livo::private_map_receipt::writeAtomic(receipt, fields, map_path);
    if (!syncFilePath(receipt.string()) || !syncDirectoryPath(directory.string()))
      throw std::runtime_error("private receipt sync failed");
    ROS_WARN_STREAM("[ PrivateMapReceipt ] exported " << receipt.string());
  }
  catch (const std::exception &error)
  {
    // Evidence failure does not change the scientific acceptance decision.
    ROS_ERROR_STREAM("[ PrivateMapReceipt ] failed: " << error.what());
  }
}

void LIVMapper::maybeExportTypedEventMapSnapshot(
    const char *route, const std::uint64_t event_sequence)
{
  if (!typed_event_map_snapshot_enabled || typed_event_map_snapshot_written ||
      typed_event_snapshot_closed || incumbent_rebuild_trajectory_active)
    return;
  if (LidarMeasures.measures.empty()) return;
  const int sensor = static_cast<int>(LidarMeasures.lio_vio_flg);
  const char *event_type = sensor == static_cast<int>(LIO) ? "LIO" :
                           sensor == static_cast<int>(VIO) ? "VIO" : nullptr;
  if (event_type == nullptr) return;
  const auto &latest = LidarMeasures.measures.back();
  const double relative_time = (sensor == static_cast<int>(VIO) ?
                                latest.vio_time : latest.lio_time) -
                               _first_lidar_time;
  if (!fast_livo::typed_event_map::typedEventMatches(
          event_type, relative_time, typed_event_map_snapshot_event_type,
          typed_event_map_snapshot_relative_time))
    return;
  if (typed_event_map_snapshot_branch_role == "identity_replay")
  {
    if (std::string(route) != "identity_replay" || !transaction_replay_active)
      return;
  }
  else if (std::string(route) != "direct" || transaction_replay_active)
    return;
  try
  {
    std::string checkpoint_sha = private_candidate_checkpoint_sha256;
    std::string journal_path = private_fork_sealed_journal_path;
    std::string journal_sha = typed_event_map_snapshot_journal_sha256;
    std::string journal_scope = "sealed_readonly";
    if (typed_event_map_snapshot_branch_role == "direct")
    {
      checkpoint_sha.clear();
      for (const std::string &path : recovery_handoff_export_files)
      {
        const auto header = readRecoveryHandoffHeader(path);
        const double completed =
            fast_livo::typed_event_map::completedTimeFromBoundaryLabel(
                header.boundary_relative_time);
        if (!std::isfinite(completed) ||
            std::abs(completed - typed_event_map_snapshot_checkpoint_completed) >
                1e-9)
          continue;
        if (!checkpoint_sha.empty())
          throw std::runtime_error("ambiguous checkpoint for typed snapshot");
        checkpoint_sha = fast_livo::source_revision::sha256File(path);
      }
      journal_path = replay_payload_journal_path;
      journal_sha = "pending_source_seal";
      journal_scope = "open_at_snapshot";
    }
    if (!fast_livo::source_revision::validSha256(checkpoint_sha))
      throw std::runtime_error("typed snapshot checkpoint hash is missing");
    if (journal_path.empty())
      throw std::runtime_error("typed snapshot journal path is missing");
    std::ostringstream name;
    name << typed_event_map_snapshot_branch_role << '_' << event_type
         << "_seq_" << event_sequence;
    const auto directory =
        std::filesystem::path(typed_event_map_snapshot_directory) / name.str();
    std::filesystem::create_directories(directory.parent_path());
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("typed snapshot directory already exists");
    PointCloudXYZI cloud;
    const auto stats = voxelmap_manager->ExportRepresentativeCloud(
        cloud, rejected_candidate_map_export_max_points,
        rejected_candidate_map_export_samples_per_plane);
    const auto map_path = directory / "typed_event_map.pcd";
    if (cloud.empty() || pcl::PCDWriter().writeBinary(map_path.string(), cloud) != 0)
      throw std::runtime_error("typed event map write failed");
    auto fields = fast_livo::private_map_receipt::runtimeHashes();
    const auto number = [](double value) {
      std::ostringstream out;
      out << std::setprecision(17) << value;
      return out.str();
    };
    fields["branch_role"] = typed_event_map_snapshot_branch_role;
    fields["event_type"] = event_type;
    fields["event_sequence"] = std::to_string(event_sequence);
    fields["relative_time_sec"] = number(relative_time);
    fields["lio_time"] = number(latest.lio_time);
    fields["vio_time"] = number(latest.vio_time);
    fields["checkpoint_sha256"] = checkpoint_sha;
    fields["checkpoint_completed_relative_sec"] = number(
        typed_event_map_snapshot_branch_role == "direct" ?
            typed_event_map_snapshot_checkpoint_completed :
            map_transaction_checkpoint.last_completed_event_relative_time);
    fields["journal_path"] = journal_path;
    fields["journal_sha256"] = journal_sha;
    fields["journal_hash_scope"] = journal_scope;
    fields["maximum_points"] = std::to_string(
        rejected_candidate_map_export_max_points);
    fields["samples_per_plane"] = std::to_string(
        rejected_candidate_map_export_samples_per_plane);
    fields["emitted_points"] = std::to_string(stats.emitted_points);
    fields["truncated"] = stats.truncated ? "1" : "0";
    if (!syncFilePath(map_path.string()))
      throw std::runtime_error("typed event map sync failed");
    const auto receipt = directory / "map.receipt";
    fast_livo::typed_event_map::writeAtomic(receipt, fields, map_path);
    if (!syncFilePath(receipt.string()) || !syncDirectoryPath(directory.string()))
      throw std::runtime_error("typed event receipt sync failed");
    typed_event_map_snapshot_written = true;
    ROS_WARN_STREAM("[ TypedEventMap ] exported " << receipt.string());
  }
  catch (const std::exception &error)
  {
    typed_event_snapshot_closed = true;
    ROS_ERROR_STREAM("[ TypedEventMap ] failed: " << error.what());
  }
}

void LIVMapper::exportRejectedCandidateMapEvidence(
    const std::string &reason, const double relative_time_sec)
{
  if (!rejected_candidate_map_export_enabled) return;
  try
  {
    std::ostringstream suffix;
    suffix << std::fixed << std::setprecision(6) << relative_time_sec;
    std::string suffix_text = suffix.str();
    std::replace(suffix_text.begin(), suffix_text.end(), '.', 'p');
    if (!suffix_text.empty() && suffix_text.front() == '-')
      suffix_text.front() = 'm';
    const std::filesystem::path directory =
        std::filesystem::path(rejected_candidate_map_export_directory) /
        ("candidate_rejected_" + suffix_text);
    std::error_code directory_error;
    std::filesystem::create_directories(directory, directory_error);
    if (directory_error)
      throw std::runtime_error(
          "cannot create rejected candidate map directory: " +
          directory_error.message());

    PointCloudXYZI representative_map;
    const auto statistics = voxelmap_manager->ExportRepresentativeCloud(
        representative_map, rejected_candidate_map_export_max_points,
        rejected_candidate_map_export_samples_per_plane);
    const std::filesystem::path map_path = directory / "candidate_map.pcd";
    pcl::PCDWriter writer;
    const bool map_ok = !representative_map.empty() &&
        writer.writeBinary(map_path.string(), representative_map) == 0;
    const std::filesystem::path state_path = directory / "candidate_state.txt";
    std::ofstream state(state_path, std::ios::out | std::ios::trunc);
    state << std::setprecision(17)
          << "schema=1\n"
          << "evaluation_only=1\n"
          << "public_commit=0\n"
          << "relative_time_sec=" << relative_time_sec << "\n"
          << "reason=" << reason << "\n"
          << "position=" << _state.pos_end.transpose() << "\n"
          << "velocity=" << _state.vel_end.transpose() << "\n"
          << "gravity=" << _state.gravity.transpose() << "\n"
          << "voxel_root_count=" << statistics.root_voxels << "\n"
          << "terminal_node_count=" << statistics.terminal_nodes << "\n"
          << "plane_node_count=" << statistics.plane_nodes << "\n"
          << "fallback_node_count=" << statistics.fallback_nodes << "\n"
          << "emitted_point_count=" << statistics.emitted_points << "\n"
          << "maximum_point_count=" << statistics.maximum_points << "\n"
          << "truncated=" << static_cast<int>(statistics.truncated) << "\n"
          << "map_path=" << map_path.string() << "\n";
    state.flush();
    state.close();
    const std::filesystem::path marker_path = directory / "export.marker";
    std::ofstream marker(marker_path, std::ios::out | std::ios::trunc);
    marker << "schema=1\n"
           << "evaluation_only=1\n"
           << "public_commit=0\n"
           << "map_write_ok=" << static_cast<int>(map_ok) << "\n"
           << "map_path=" << map_path.string() << "\n"
           << "state_path=" << state_path.string() << "\n"
           << "relative_time_sec=" << std::setprecision(17)
           << relative_time_sec << "\n"
           << "reason=" << reason << "\n"
           << "root_voxels=" << statistics.root_voxels << "\n"
           << "emitted_points=" << statistics.emitted_points << "\n"
           << "truncated=" << static_cast<int>(statistics.truncated) << "\n";
    marker.flush();
    marker.close();
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "evaluation_only=1;public_commit=0;map_write_ok="
           << static_cast<int>(map_ok)
           << ";directory=" << directory.string()
           << ";root_voxels=" << statistics.root_voxels
           << ";emitted_points=" << statistics.emitted_points
           << ";truncated=" << static_cast<int>(statistics.truncated);
    publishTransactionStatus("candidate_rejected_map_exported", detail.str(),
                             relative_time_sec);
    ROS_WARN_STREAM("[ MapExport ] rejected candidate evidence directory="
                    << directory.string() << " roots="
                    << statistics.root_voxels << " points="
                    << statistics.emitted_points << " truncated="
                    << statistics.truncated << " map_ok=" << map_ok);
  }
  catch (const std::exception &error)
  {
    publishTransactionStatus(
        "candidate_rejected_map_export_failed",
        std::string("evaluation_only=1;public_commit=0;error=") +
            error.what(), relative_time_sec);
    ROS_ERROR_STREAM("[ MapExport ] rejected candidate export failed: "
                     << error.what());
  }
}

void LIVMapper::saveTrajectoryPoses()
{
  if (!pose_output_en || trajectory_poses.empty()) return;
  const std::string path = std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt";
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  if (!output.is_open())
  {
    ROS_ERROR_STREAM("[ Transaction ] Cannot write trajectory: " << path);
    return;
  }
  output << std::fixed;
  for (const auto &entry : trajectory_poses)
  {
    output << entry.first;
    for (const double value : entry.second) output << " " << value;
    output << std::endl;
  }
}

LidarMeasureGroup LIVMapper::cloneMeasureGroup(const LidarMeasureGroup &source) const
{
  LidarMeasureGroup clone;
  clone.lidar_frame_beg_time = source.lidar_frame_beg_time;
  clone.lidar_frame_end_time = source.lidar_frame_end_time;
  clone.last_lio_update_time = source.last_lio_update_time;
  clone.lio_vio_flg = source.lio_vio_flg;
  clone.lidar_scan_index_now = source.lidar_scan_index_now;
  clone.raw_reprocess_begin_time = source.raw_reprocess_begin_time;
  clone.raw_reprocess_end_time = source.raw_reprocess_end_time;
  clone.raw_reprocess_recipe_valid = source.raw_reprocess_recipe_valid;
  clone.raw_reprocess_split_by_time = source.raw_reprocess_split_by_time;
  clone.raw_preprocess_lidar_type = source.raw_preprocess_lidar_type;
  clone.raw_preprocess_point_filter_num = source.raw_preprocess_point_filter_num;
  clone.raw_preprocess_scan_lines = source.raw_preprocess_scan_lines;
  clone.raw_preprocess_blind = source.raw_preprocess_blind;
  clone.raw_preprocess_feature_enabled = source.raw_preprocess_feature_enabled;
  if (raw_lidar_reprocess_enabled &&
      (source.lio_vio_flg == LIO || source.lio_vio_flg == LO))
    clone.raw_lidar_frames = source.raw_lidar_frames;

  // With IMU enabled Process2()/UndistortPcl() consume pcl_proc_cur only; the
  // raw lidar and carry-over pcl_proc_next are synchronization scratch owned
  // by the live ingest path.  Deep-copying all three clouds for every VIO/LIO
  // event retained multiple versions of the same scan and dominated replay
  // memory.  Keep the exact processed cloud required for deterministic replay.
  // LiDAR-only operation still needs the raw scan in Forward_without_imu().
  if (!imu_en && source.lidar != nullptr)
    clone.lidar.reset(new PointCloudXYZI(*source.lidar));
  else
    clone.lidar.reset(new PointCloudXYZI());
  clone.pcl_proc_cur.reset(source.pcl_proc_cur == nullptr ? new PointCloudXYZI() : new PointCloudXYZI(*source.pcl_proc_cur));
  clone.pcl_proc_next.reset(new PointCloudXYZI());

  clone.measures.clear();
  for (const MeasureGroup &source_measure : source.measures)
  {
    MeasureGroup measure;
    measure.vio_time = source_measure.vio_time;
    measure.lio_time = source_measure.lio_time;
    measure.img = source_measure.img.clone();
    // ROS IMU inputs are immutable ConstPtr messages. Sharing their ownership
    // is exact and avoids duplicating the same high-rate samples across the
    // adjacent VIO and LIO event groups that reference them.
    for (const sensor_msgs::Imu::ConstPtr &source_imu : source_measure.imu)
      if (source_imu != nullptr) measure.imu.push_back(source_imu);
    clone.measures.push_back(std::move(measure));
  }
  return clone;
}

void LIVMapper::initializeRawReplayRecipe(LidarMeasureGroup &group,
                                          const double begin_time,
                                          const double end_time,
                                          const bool split_by_time)
{
  group.raw_lidar_frames.clear();
  group.raw_reprocess_begin_time = begin_time;
  group.raw_reprocess_end_time = end_time;
  group.raw_reprocess_recipe_valid = raw_lidar_reprocess_enabled;
  group.raw_reprocess_split_by_time = split_by_time;
  group.raw_preprocess_lidar_type = p_pre->lidar_type;
  group.raw_preprocess_point_filter_num = p_pre->point_filter_num;
  group.raw_preprocess_scan_lines = p_pre->N_SCANS;
  group.raw_preprocess_blind = p_pre->blind;
  group.raw_preprocess_feature_enabled = p_pre->feature_enabled;
}

bool LIVMapper::rebuildProcessedCloudFromRaw(
    LidarMeasureGroup &group, const double event_relative_time,
    std::size_t &corrected_points,
    const double diagnostic_inverse_scale)
{
  corrected_points = 0;
  const bool diagnostic_only = std::isfinite(diagnostic_inverse_scale);
  if (!raw_lidar_reprocess_enabled) return !diagnostic_only;
  if (!group.raw_reprocess_recipe_valid || group.raw_lidar_frames.empty() ||
      !std::isfinite(group.raw_reprocess_begin_time) ||
      !std::isfinite(group.raw_reprocess_end_time) ||
      group.raw_reprocess_end_time < group.raw_reprocess_begin_time)
  {
    if (!diagnostic_only) ++raw_lidar_reprocess_failures;
    return false;
  }

  if (!replay_preprocessor) replay_preprocessor.reset(new Preprocess());
  replay_preprocessor->set(group.raw_preprocess_feature_enabled,
                           group.raw_preprocess_lidar_type,
                           group.raw_preprocess_blind,
                           group.raw_preprocess_point_filter_num);
  replay_preprocessor->N_SCANS = group.raw_preprocess_scan_lines;
  replay_preprocessor->blind_sqr =
      group.raw_preprocess_blind * group.raw_preprocess_blind;

  PointCloudXYZI::Ptr rebuilt(new PointCloudXYZI());
  fast_livo::audit::Digest raw_digest;
  fast_livo::audit::Digest corrected_digest;
  std::size_t event_raw_bytes = 0;
  try
  {
    for (const RawLidarFrame &frame : group.raw_lidar_frames)
    {
      raw_digest.scalar(frame.serialized_checksum);
      event_raw_bytes += frame.serialized_message.size();
      PointCloudXYZI::Ptr processed(new PointCloudXYZI());
      std::uint64_t corrected_checksum = 0;
      const bool profile_active = !diagnostic_only && private_lidar_profile_enabled;
      const bool estimated_correction_active = !diagnostic_only && !profile_active &&
          estimated_range_scale_candidate_enabled &&
          estimated_range_scale_candidate_plan.appliesAt(
              frame.oracle_source_relative_time);
      const auto oracle_correction = !diagnostic_only && !profile_active && !estimated_correction_active ?
          fast_livo::replay_oracle::rawScanCorrection(
              private_raw_scan_oracle_config, frame.oracle_source_relative_time,
              frame.oracle_range_scale_active, frame.oracle_inverse_range_scale) :
          fast_livo::replay_oracle::RawScanCorrection{};
      const bool oracle_correction_active = !diagnostic_only &&
          !estimated_correction_active && oracle_correction.active;
      bool correction_active = diagnostic_only ?
          diagnostic_inverse_scale != 1.0 :
          (estimated_correction_active || oracle_correction_active);
      double inverse_range_scale = diagnostic_only ?
          diagnostic_inverse_scale : estimated_correction_active ?
          estimated_range_scale_candidate_plan.inverseScaleAt(
              frame.oracle_source_relative_time) :
          oracle_correction.inverse_scale;
      if (frame.message_type == RawLidarMessageType::POINTCLOUD2)
      {
        sensor_msgs::PointCloud2 message =
            fast_livo::raw_replay::decode<sensor_msgs::PointCloud2>(frame);
        if (profile_active)
        {
          inverse_range_scale = private_lidar_profile.consume(
              message.header.stamp.toNSec(), frame.serialized_message, p2SensorStampNs(group));
          correction_active = inverse_range_scale != 1.0;
        }
        if (correction_active)
          corrected_points += fast_livo::raw_replay::applyRangeScale(
              message, inverse_range_scale);
        const RawLidarFrame corrected = fast_livo::raw_replay::capture(
            message, frame.message_type, frame.effective_header_time,
            PointCloudXYZI::Ptr(), frame.oracle_source_relative_time,
            inverse_range_scale, correction_active);
        corrected_checksum = corrected.serialized_checksum;
        sensor_msgs::PointCloud2::ConstPtr input(
            new sensor_msgs::PointCloud2(std::move(message)));
        replay_preprocessor->process(input, processed);
      }
      else if (frame.message_type == RawLidarMessageType::LIVOX_CUSTOM)
      {
        livox_ros_driver::CustomMsg message =
            fast_livo::raw_replay::decode<livox_ros_driver::CustomMsg>(frame);
        if (profile_active)
        {
          inverse_range_scale = private_lidar_profile.consume(
              message.header.stamp.toNSec(), frame.serialized_message, p2SensorStampNs(group));
          correction_active = inverse_range_scale != 1.0;
        }
        if (correction_active)
          corrected_points += fast_livo::raw_replay::applyRangeScale(
              message, inverse_range_scale);
        const RawLidarFrame corrected = fast_livo::raw_replay::capture(
            message, frame.message_type, frame.effective_header_time,
            PointCloudXYZI::Ptr(), frame.oracle_source_relative_time,
            inverse_range_scale, correction_active);
        corrected_checksum = corrected.serialized_checksum;
        livox_ros_driver::CustomMsg::ConstPtr input(
            new livox_ros_driver::CustomMsg(std::move(message)));
        replay_preprocessor->process(input, processed);
      }
      else
        throw std::runtime_error("unsupported raw LiDAR message type");
      corrected_digest.scalar(corrected_checksum);

      if (!diagnostic_only && replay_range_scale_oracle_config.enabled)
      {
        for (PointType &point : processed->points)
        {
          point.normal_x = static_cast<float>(frame.oracle_inverse_range_scale);
          point.normal_y = fast_livo::replay_oracle::kRangeScaleProvenanceMarker;
          point.normal_z = static_cast<float>(
              frame.oracle_source_relative_time);
        }
      }

      if (!group.raw_reprocess_split_by_time)
      {
        rebuilt->points.insert(rebuilt->points.end(), processed->points.begin(),
                               processed->points.end());
        continue;
      }
      const float minimum_offset_ms = static_cast<float>(
          (group.raw_reprocess_begin_time - frame.effective_header_time) *
          1000.0);
      const float maximum_offset_ms = static_cast<float>(
          (group.raw_reprocess_end_time - frame.effective_header_time) *
          1000.0);
      for (const PointType &source : processed->points)
      {
        if (source.curvature < minimum_offset_ms ||
            source.curvature >= maximum_offset_ms)
          continue;
        PointType point = source;
        // Match sync_packages() exactly: float curvature is promoted to double
        // by the double timestamp delta, then rounded once on assignment.
        // Rounding the offset to float before addition changes the cloud hash.
        point.curvature = fast_livo::raw_replay::rebaseCurvatureLikeOnline(
            point.curvature, frame.effective_header_time,
            group.raw_reprocess_begin_time);
        rebuilt->points.push_back(point);
      }
    }
  }
  catch (const std::exception &error)
  {
    if (!diagnostic_only) ++raw_lidar_reprocess_failures;
    if (!diagnostic_only && raw_lidar_reprocess_audit.is_open())
      raw_lidar_reprocess_audit << event_relative_time << ','
          << group.raw_lidar_frames.size() << ',' << event_raw_bytes << ','
          << raw_digest.hex() << ',' << corrected_digest.hex() << ','
          << corrected_points << ",0,,failure:" << error.what() << '\n';
    ROS_ERROR_STREAM("[ RawReplay ] " << error.what());
    return false;
  }

  rebuilt->width = rebuilt->points.size();
  rebuilt->height = 1;
  rebuilt->is_dense = false;
  group.pcl_proc_cur = rebuilt;
  if (!imu_en) group.lidar = rebuilt;
  group.pcl_proc_next.reset(new PointCloudXYZI());
  if (!diagnostic_only)
  {
    ++raw_lidar_reprocess_events;
    raw_lidar_reprocess_frames += group.raw_lidar_frames.size();
    raw_lidar_reprocess_bytes += event_raw_bytes;
    raw_lidar_reprocess_corrected_points += corrected_points;
    raw_lidar_reprocess_output_points += rebuilt->size();
  }
  if (!diagnostic_only && raw_lidar_reprocess_audit.is_open())
  {
    raw_lidar_reprocess_audit << event_relative_time << ','
        << group.raw_lidar_frames.size() << ',' << event_raw_bytes << ','
        << raw_digest.hex() << ',' << corrected_digest.hex() << ','
        << corrected_points << ',' << rebuilt->size() << ','
        << fast_livo::audit::cloudDigest(rebuilt) << ",ok\n";
    raw_lidar_reprocess_audit.flush();
  }
  return true;
}

void LIVMapper::runPrivateFrozenScaleProbe(
    const double relative_time, const StatesGroup &state_before_imu,
    const ImuProcess::Checkpoint &imu_before_event)
{
  if (!private_frozen_scale_probe_map_frozen ||
      (private_frozen_scale_probe_snapshot_reference_enabled &&
       !private_frozen_scale_probe_reference_snapshot) ||
      !private_frozen_scale_probe_csv.is_open()) return;
  std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] enter t=%.6f raw_frames=%zu\n",
               relative_time, LidarMeasures.raw_lidar_frames.size());
  std::fflush(stderr);
  const std::string source_digest =
      fast_livo::audit::cloudDigest(feats_undistort);
  PointCloudXYZI::Ptr common_identity_sample;
  PointCloudXYZI::Ptr common_identity_input_sample;
  std::vector<std::uint64_t> common_identity_keys;
  StatesGroup common_identity_state = state_before_imu;
  // Diagnostic-only bank: each scale is evaluated from the identical original
  // raw message, IMU pre-event state, and frozen pre-onset map. No branch state
  // or public publication is changed by a score.
  for (int scale_index = 0; scale_index <= 4; ++scale_index)
  {
    const double scale = 1.0 + 0.01 * scale_index;
    std::size_t corrected_points = 0;
    std::size_t downsampled_points = 0;
    VoxelMapManager::ResidualStatistics residual;
    StatesGroup probe_state = state_before_imu;
    std::string probe_digest;
    std::string status = "ok";
    bool identity_equal = false;
    int pose_refinement_steps = 0;
    std::size_t initial_matched_points = 0;
    double initial_residual = std::numeric_limits<double>::infinity();
    double pose_shift_m = 0.0;
    VoxelMapManager::CommonPlaneStatistics common_planes;
    std::string common_status = "disabled";
    try
    {
      std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] before raw t=%.6f scale=%.2f\n",
                   relative_time, scale);
      std::fflush(stderr);
      // The default copy aliases live PCL scratch buffers. Replay must own a
      // deep-cloned measure group before a second preprocessing/IMU pass.
      LidarMeasureGroup probe_group = cloneMeasureGroup(LidarMeasures);
      if (!rebuildProcessedCloudFromRaw(
              probe_group, relative_time, corrected_points, 1.0 / scale))
        throw std::runtime_error("raw_reprocess_failed");
      std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] after raw t=%.6f scale=%.2f\n",
                   relative_time, scale);
      std::fflush(stderr);
      // A private IMU processor avoids mutating or restoring the incumbent's
      // live undistortion buffers. A prior diagnostic crashed when a second
      // pass on the live processor was restored after scoring.
      ImuProcess probe_imu;
      probe_imu.set_extrinsic(extT, extR);
      probe_imu.set_gyr_cov_scale(p_imu->cov_gyr);
      probe_imu.set_acc_cov_scale(p_imu->cov_acc);
      probe_imu.set_inv_expo_cov(p_imu->cov_inv_expo);
      probe_imu.set_gyr_bias_cov(p_imu->cov_bias_gyr);
      probe_imu.set_acc_bias_cov(p_imu->cov_bias_acc);
      probe_imu.set_imu_init_frame_num(imu_int_frame);
      if (!imu_en) probe_imu.disable_imu();
      if (!gravity_est_en) probe_imu.disable_gravity_est();
      if (!ba_bg_est_en) probe_imu.disable_bias_est();
      if (!exposure_estimate_en) probe_imu.disable_exposure_est();
      probe_imu.first_lidar_time = _first_lidar_time;
      probe_imu.lidar_type = p_imu->lidar_type;
      probe_imu.restoreCheckpoint(imu_before_event);
      std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] before imu t=%.6f scale=%.2f\n",
                   relative_time, scale);
      std::fflush(stderr);
      PointCloudXYZI::Ptr probe_undistorted(new PointCloudXYZI());
      probe_imu.Process2(probe_group, probe_state, probe_undistorted);
      std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] after imu t=%.6f scale=%.2f\n",
                   relative_time, scale);
      std::fflush(stderr);
      probe_digest = fast_livo::audit::cloudDigest(probe_undistorted);
      identity_equal = scale_index == 0 && probe_digest == source_digest;
      PointCloudXYZI::Ptr downsampled(new PointCloudXYZI());
      // The distro PCL VoxelGrid instantiation and this native-optimized
      // translation unit may use different Eigen alignment defaults.  Keep
      // ownership of the output point buffer on our side: a voxel filter
      // cannot emit more points than its input, so it only resizes in-place.
      downsampled->points.reserve(probe_undistorted->size());
      pcl::VoxelGrid<PointType> filter;
      filter.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                         filter_size_surf_min);
      filter.setInputCloud(probe_undistorted);
      filter.filter(*downsampled);
      downsampled_points = downsampled->size();
      if (private_frozen_scale_probe_pose_refinement_enabled)
      {
        const auto refinement = voxelmap_manager->RefinePoseAgainstSnapshot(
            *private_frozen_scale_probe_reference_snapshot,
            probe_state, downsampled);
        residual = refinement.final;
        initial_matched_points = refinement.initial.matched_points;
        initial_residual = refinement.initial.mean_abs_residual;
        pose_shift_m = (refinement.state.pos_end - probe_state.pos_end).norm();
        pose_refinement_steps = refinement.accepted_steps;
        probe_state = refinement.state;
      }
      else
      {
        residual = private_frozen_scale_probe_snapshot_reference_enabled
            ? voxelmap_manager->EvaluateResidualsAgainstSnapshot(
                  *private_frozen_scale_probe_reference_snapshot,
                  probe_state, downsampled)
            : voxelmap_manager->EvaluateResidualsAgainstCurrentMap(
                  probe_state, downsampled);
        initial_matched_points = residual.matched_points;
        initial_residual = residual.mean_abs_residual;
      }
      if (private_frozen_scale_probe_common_support_enabled)
      {
        common_status = "missing_shared_points";
        const PointCloudXYZI::Ptr &input = probe_group.pcl_proc_cur;
        if (input && !probe_undistorted->empty() &&
            input->size() == probe_undistorted->size())
        {
          const auto key = [](const PointType &point) {
            std::uint32_t time_bits = 0;
            std::uint32_t intensity_bits = 0;
            std::memcpy(&time_bits, &point.curvature, sizeof(time_bits));
            std::memcpy(&intensity_bits, &point.intensity,
                        sizeof(intensity_bits));
            return (static_cast<std::uint64_t>(time_bits) << 32) |
                   intensity_bits;
          };
          const std::size_t duplicate = std::numeric_limits<std::size_t>::max();
          std::unordered_map<std::uint64_t, std::size_t> by_metadata;
          by_metadata.reserve(input->size());
          for (std::size_t index = 0; index < input->size(); ++index)
          {
            const auto [item, inserted] =
                by_metadata.emplace(key(input->points[index]), index);
            if (!inserted) item->second = duplicate;
          }
          if (scale_index == 0)
          {
            common_identity_sample.reset(new PointCloudXYZI());
            common_identity_input_sample.reset(new PointCloudXYZI());
            const std::size_t count =
                std::min<std::size_t>(2048, probe_undistorted->size());
            common_identity_sample->points.reserve(count);
            common_identity_input_sample->points.reserve(count);
            common_identity_keys.reserve(count);
            for (std::size_t sample_index = 0; sample_index < count;
                 ++sample_index)
            {
              const std::size_t index = sample_index * input->size() / count;
              const PointType &point = input->points[index];
              if (point.x * point.x + point.y * point.y +
                      point.z * point.z < 2.25f ||
                  by_metadata.at(key(point)) != index)
                continue;
              common_identity_keys.push_back(key(point));
              common_identity_input_sample->points.push_back(point);
              common_identity_sample->points.push_back(
                  probe_undistorted->points[index]);
            }
            common_identity_state = probe_state;
          }
          if (common_identity_sample && !common_identity_sample->empty())
          {
            PointCloudXYZI::Ptr sample(new PointCloudXYZI());
            sample->points.reserve(common_identity_keys.size());
            bool aligned = true;
            for (std::size_t index = 0;
                 index < common_identity_keys.size(); ++index)
            {
              const auto item = by_metadata.find(common_identity_keys[index]);
              if (item == by_metadata.end() || item->second == duplicate)
              {
                aligned = false;
                break;
              }
              const PointType &reference =
                  common_identity_input_sample->points[index];
              const PointType &candidate = input->points[item->second];
              const Eigen::Vector3d ray_reference(reference.x, reference.y,
                                                   reference.z);
              const Eigen::Vector3d ray_candidate(candidate.x, candidate.y,
                                                   candidate.z);
              if (ray_reference.squaredNorm() == 0.0 ||
                  ray_candidate.squaredNorm() == 0.0 ||
                  ray_reference.normalized().dot(
                      ray_candidate.normalized()) < 0.9999)
              {
                aligned = false;
                break;
              }
              sample->points.push_back(
                  probe_undistorted->points[item->second]);
            }
            common_status = aligned ? "insufficient_reference_matches" :
                "unmatched_original_point_identity";
            if (aligned)
            {
              common_planes =
                  voxelmap_manager->EvaluateCommonPlanesAgainstSnapshot(
                      *private_frozen_scale_probe_reference_snapshot,
                      common_identity_state, common_identity_sample,
                      probe_state, sample);
              if (common_planes.valid) common_status = "ok";
            }
          }
        }
      }
      std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] after score t=%.6f scale=%.2f\n",
                   relative_time, scale);
      std::fflush(stderr);
      if (!residual.valid) status = "insufficient_reference_matches";
    }
    catch (const std::exception &error)
    {
      status = std::string("error:") + error.what();
    }
    private_frozen_scale_probe_csv
        << private_frozen_scale_probe_reference_sec << ',' << relative_time
        << ',' << scale << ',' << LidarMeasures.raw_lidar_frames.size()
        << ',' << corrected_points << ',' << downsampled_points << ','
        << residual.matched_points << ',' << residual.match_ratio << ','
        << residual.mean_abs_residual << ',' << static_cast<int>(identity_equal)
        << ',' << source_digest << ',' << probe_digest << ','
        << probe_state.pos_end.x() << ',' << probe_state.pos_end.y() << ','
        << probe_state.pos_end.z() << ',' << status << ','
        << pose_refinement_steps << ',' << initial_matched_points << ','
        << initial_residual << ',' << pose_shift_m << ','
        << common_planes.shared_points << ','
        << common_planes.reference_matches << ','
        << common_planes.reference_mean_abs_residual << ','
        << common_planes.candidate_mean_abs_residual << ','
        << common_planes.mean_world_displacement << ','
        << common_status << '\n';
  }
  private_frozen_scale_probe_csv.flush();
}

void LIVMapper::resetReplayPayloadJournal()
{
  if (!disk_backed_replay_payload_enabled) return;
  replay_payload_journal.reset();
}

void LIVMapper::spillReplayPayload(BufferedTransactionEvent &event)
{
  if (!disk_backed_replay_payload_enabled) return;
  event.payload_record = replay_payload_journal.spill(event.measures);
}

LidarMeasureGroup LIVMapper::materializeReplayEvent(
    const BufferedTransactionEvent &event)
{
  if (!event.payload_record.spilled) return cloneMeasureGroup(event.measures);
  return replay_payload_journal.materialize(event.measures, event.payload_record);
}

LIVMapper::RecoveryReplayCoverage LIVMapper::assessRecoveryReplayCoverage(
    const double checkpoint_relative_time,
    const std::deque<BufferedTransactionEvent> &events,
    const double maximum_gap_sec)
{
  RecoveryReplayCoverage coverage;
  if (!std::isfinite(checkpoint_relative_time) ||
      !std::isfinite(maximum_gap_sec) || maximum_gap_sec <= 0.0)
    return coverage;

  double previous_time = -std::numeric_limits<double>::infinity();
  for (const BufferedTransactionEvent &event : events)
  {
    if (!std::isfinite(event.relative_time) ||
        event.relative_time < previous_time)
      return coverage;
    previous_time = event.relative_time;
    if (event.relative_time <= checkpoint_relative_time) continue;
    coverage.first_event_relative_time = event.relative_time;
    coverage.gap_sec = event.relative_time - checkpoint_relative_time;
    coverage.continuous = coverage.gap_sec <= maximum_gap_sec;
    return coverage;
  }
  return coverage;
}

LIVMapper::MapperCheckpoint LIVMapper::captureMapperCheckpoint() const
{
  MapperCheckpoint checkpoint;
  checkpoint.state = _state;
  checkpoint.propagated_state = state_propagat;
  checkpoint.latest_ekf_state = latest_ekf_state;
  checkpoint.imu_propagate = imu_propagate;
  checkpoint.imu_process = p_imu->createCheckpoint();
  checkpoint.point_variances = _pv_list;
  checkpoint.lidar_world_for_visual = *pcl_w_wait_pub;
  checkpoint.pending_publish_cloud = *pcl_wait_pub;
  checkpoint.voxel_position_last = voxelmap_manager->position_last_;
  checkpoint.voxel_last_slide_position = voxelmap_manager->last_slide_position;
  checkpoint.latest_ekf_time = latest_ekf_time;
  checkpoint.last_completed_event_relative_time =
      last_completed_event_relative_time;
  checkpoint.average_processing_time = aver_time_consu;
  checkpoint.path_pose_count = path.poses.size();
  checkpoint.trajectory_pose_count = trajectory_poses.size();
  checkpoint.pcd_rgb_point_count = pcl_wait_save->points.size();
  checkpoint.pcd_intensity_point_count = pcl_wait_save_intensity->points.size();
  checkpoint.voxel_frame_id = voxelmap_manager->current_frame_id_;
  checkpoint.voxel_next_plane_id = voxelmap_manager->nextPlaneId();
  checkpoint.visual_frame_counter = Frame::frame_counter_;
  checkpoint.visual_processed_frames = vio_manager->frame_count;
  checkpoint.processed_lidar_frames = frame_num;
  checkpoint.lidar_map_inited = lidar_map_inited;
  checkpoint.ekf_finish_once = ekf_finish_once;
  checkpoint.state_update_flg = state_update_flg;
  return checkpoint;
}

std::vector<std::uint8_t> LIVMapper::serializeMapperCheckpoint(
    const MapperCheckpoint &checkpoint)
{
  Writer writer;
  writer.pod(kMapperCheckpointMagic);
  writer.pod(kMapperCheckpointSchemaVersion);
  writeState(writer, checkpoint.state);
  writeState(writer, checkpoint.propagated_state);
  writeState(writer, checkpoint.latest_ekf_state);
  writeState(writer, checkpoint.imu_propagate);
  const std::vector<std::uint8_t> imu_payload =
      ImuProcess::serializeCheckpoint(checkpoint.imu_process);
  writer.pod(static_cast<std::uint64_t>(imu_payload.size()));
  writer.raw(imu_payload.data(), imu_payload.size());
  writer.pod(static_cast<std::uint64_t>(checkpoint.point_variances.size()));
  for (const pointWithVar &point : checkpoint.point_variances)
    writeMapperPointWithVar(writer, point);
  writeMapperCloud(writer, checkpoint.lidar_world_for_visual);
  writeMapperCloud(writer, checkpoint.pending_publish_cloud);
  writer.eigen(checkpoint.voxel_position_last);
  writer.eigen(checkpoint.voxel_last_slide_position);
  writer.pod(checkpoint.latest_ekf_time);
  writer.pod(checkpoint.last_completed_event_relative_time);
  writer.pod(checkpoint.average_processing_time);
  writer.pod(static_cast<std::uint64_t>(checkpoint.path_pose_count));
  writer.pod(static_cast<std::uint64_t>(checkpoint.trajectory_pose_count));
  writer.pod(static_cast<std::uint64_t>(checkpoint.pcd_rgb_point_count));
  writer.pod(static_cast<std::uint64_t>(checkpoint.pcd_intensity_point_count));
  writer.pod(checkpoint.voxel_frame_id);
  writer.pod(checkpoint.visual_frame_counter);
  writer.pod(checkpoint.visual_processed_frames);
  writer.pod(checkpoint.processed_lidar_frames);
  writer.pod(static_cast<std::uint8_t>(checkpoint.lidar_map_inited));
  writer.pod(static_cast<std::uint8_t>(checkpoint.ekf_finish_once));
  writer.pod(static_cast<std::uint8_t>(checkpoint.state_update_flg));
  writer.pod(checkpoint.voxel_next_plane_id);
  const std::uint64_t checksum =
      fast_livo::checkpoint_io::fnv1a(writer.bytes());
  writer.pod(checksum);
  return writer.take();
}

LIVMapper::MapperCheckpoint LIVMapper::deserializeMapperCheckpoint(
    const std::vector<std::uint8_t> &payload)
{
  if (payload.size() < sizeof(std::uint64_t))
    throw std::runtime_error("truncated mapper checkpoint checksum");
  std::uint64_t stored_checksum = 0;
  std::memcpy(&stored_checksum,
              payload.data() + payload.size() - sizeof(stored_checksum),
              sizeof(stored_checksum));
  const std::vector<std::uint8_t> body(
      payload.begin(), payload.end() - sizeof(stored_checksum));
  if (fast_livo::checkpoint_io::fnv1a(body) != stored_checksum)
    throw std::runtime_error("mapper checkpoint checksum mismatch");
  Reader reader(body);
  if (reader.pod<std::uint64_t>() != kMapperCheckpointMagic)
    throw std::runtime_error("invalid mapper checkpoint magic");
  const auto mapper_schema = reader.pod<std::uint32_t>();
  if (mapper_schema != 1 && mapper_schema != kMapperCheckpointSchemaVersion)
    throw std::runtime_error("unsupported mapper checkpoint schema");
  MapperCheckpoint checkpoint;
  checkpoint.state = readState(reader);
  checkpoint.propagated_state = readState(reader);
  checkpoint.latest_ekf_state = readState(reader);
  checkpoint.imu_propagate = readState(reader);
  const std::uint64_t imu_size = fast_livo::checkpoint_io::checkedCount(
      reader, static_cast<std::uint64_t>(reader.remaining()),
      "mapper IMU payload byte");
  std::vector<std::uint8_t> imu_payload(static_cast<std::size_t>(imu_size));
  reader.raw(imu_payload.data(), imu_payload.size());
  checkpoint.imu_process = ImuProcess::deserializeCheckpoint(imu_payload);
  const std::uint64_t variance_count = fast_livo::checkpoint_io::checkedCount(
      reader, 1000000000, "mapper point-variance");
  checkpoint.point_variances.reserve(static_cast<std::size_t>(variance_count));
  for (std::uint64_t index = 0; index < variance_count; ++index)
    checkpoint.point_variances.push_back(readMapperPointWithVar(reader));
  checkpoint.lidar_world_for_visual = readMapperCloud(reader);
  checkpoint.pending_publish_cloud = readMapperCloud(reader);
  reader.eigen(checkpoint.voxel_position_last);
  reader.eigen(checkpoint.voxel_last_slide_position);
  checkpoint.latest_ekf_time = reader.pod<double>();
  checkpoint.last_completed_event_relative_time = reader.pod<double>();
  checkpoint.average_processing_time = reader.pod<double>();
  checkpoint.path_pose_count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
  checkpoint.trajectory_pose_count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
  checkpoint.pcd_rgb_point_count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
  checkpoint.pcd_intensity_point_count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
  checkpoint.voxel_frame_id = reader.pod<int>();
  checkpoint.visual_frame_counter = reader.pod<int>();
  checkpoint.visual_processed_frames = reader.pod<int>();
  checkpoint.processed_lidar_frames = reader.pod<int>();
  checkpoint.lidar_map_inited = reader.pod<std::uint8_t>() != 0;
  checkpoint.ekf_finish_once = reader.pod<std::uint8_t>() != 0;
  checkpoint.state_update_flg = reader.pod<std::uint8_t>() != 0;
  if (mapper_schema >= 2) checkpoint.voxel_next_plane_id = reader.pod<int>();
  if (!reader.finished())
    throw std::runtime_error("trailing bytes in mapper checkpoint");
  return checkpoint;
}

std::uint64_t LIVMapper::mapperCheckpointFingerprint(
    const MapperCheckpoint &checkpoint)
{
  return fast_livo::checkpoint_io::fnv1a(
      serializeMapperCheckpoint(checkpoint));
}

std::vector<std::uint8_t> LIVMapper::serializeRecoveryHandoffBundle(
    const RecoveryHandoffBundle &bundle)
{
  if (bundle.generation == 0 || !std::isfinite(bundle.boundary_relative_time) ||
      bundle.boundary_relative_time < 0.0 || bundle.voxel_map == nullptr ||
      bundle.visual_map == nullptr || bundle.visual_patch_size_total < 0)
    throw std::invalid_argument("incomplete recovery handoff bundle");
  Writer writer;
  writer.pod(kRecoveryHandoffMagic);
  writer.pod(kRecoveryHandoffSchemaVersion);
  writer.pod(bundle.generation);
  writer.pod(bundle.boundary_relative_time);
  writer.pod(static_cast<std::uint64_t>(bundle.next_event_sequence));
  writer.pod(bundle.visual_patch_size_total);
  writer.string(bundle.replay_journal_path);
  writer.pod(static_cast<std::uint64_t>(bundle.replay_records.size()));
  for (const auto &record : bundle.replay_records)
  {
    writer.pod(record.offset);
    writer.pod(record.size);
    writer.pod(record.checksum);
    writer.pod(static_cast<std::uint8_t>(record.spilled));
  }
  writeDetectorSnapshot(writer, bundle.detector);
  writeBlob(writer, serializeMapperCheckpoint(bundle.mapper));
  writeBlob(writer, bundle.voxel_map->serialize());
  writeBlob(writer, VIOManager::serializeCheckpointForPatchSize(
      *bundle.visual_map, bundle.visual_patch_size_total));
  const std::uint64_t checksum =
      fast_livo::checkpoint_io::fnv1a(writer.bytes());
  writer.pod(checksum);
  return writer.take();
}

LIVMapper::RecoveryHandoffBundle LIVMapper::deserializeRecoveryHandoffBundle(
    const std::vector<std::uint8_t> &payload)
{
  return deserializeRecoveryHandoffBundle(payload.data(), payload.size());
}

LIVMapper::RecoveryHandoffBundle LIVMapper::deserializeRecoveryHandoffBundle(
    const std::uint8_t *payload, const std::size_t payload_size)
{
  if (payload_size < sizeof(std::uint64_t))
    throw std::runtime_error("truncated recovery handoff checksum");
  std::uint64_t stored_checksum = 0;
  std::memcpy(&stored_checksum,
              payload + payload_size - sizeof(stored_checksum),
              sizeof(stored_checksum));
  const std::size_t body_size = payload_size - sizeof(stored_checksum);
  if (fast_livo::checkpoint_io::fnv1a(payload, body_size) != stored_checksum)
    throw std::runtime_error("recovery handoff checksum mismatch");
  Reader reader(payload, body_size);
  if (reader.pod<std::uint64_t>() != kRecoveryHandoffMagic)
    throw std::runtime_error("invalid recovery handoff magic");
  if (reader.pod<std::uint32_t>() != kRecoveryHandoffSchemaVersion)
    throw std::runtime_error("unsupported recovery handoff schema");
  RecoveryHandoffBundle bundle;
  bundle.generation = reader.pod<std::uint64_t>();
  bundle.boundary_relative_time = reader.pod<double>();
  bundle.next_event_sequence =
      static_cast<std::size_t>(reader.pod<std::uint64_t>());
  bundle.visual_patch_size_total = reader.pod<int>();
  bundle.replay_journal_path = reader.string();
  const std::uint64_t record_count = fast_livo::checkpoint_io::checkedCount(
      reader, 100000000, "replay record");
  bundle.replay_records.resize(static_cast<std::size_t>(record_count));
  for (auto &record : bundle.replay_records)
  {
    record.offset = reader.pod<std::uint64_t>();
    record.size = reader.pod<std::uint64_t>();
    record.checksum = reader.pod<std::uint64_t>();
    record.spilled = reader.pod<std::uint8_t>() != 0;
  }
  bundle.detector = readDetectorSnapshot(reader);
  const ByteView mapper_component =
      readBlobView(reader, "mapper component byte");
  bundle.mapper = deserializeMapperCheckpoint(std::vector<std::uint8_t>(
      mapper_component.data, mapper_component.data + mapper_component.size));
  const ByteView voxel_component =
      readBlobView(reader, "voxel component byte");
  bundle.voxel_map = VoxelMapSnapshot::deserialize(
      voxel_component.data, voxel_component.size);
  const ByteView visual_component =
      readBlobView(reader, "visual component byte");
  bundle.visual_map = VIOManager::deserializeCheckpointForPatchSize(
      visual_component.data, visual_component.size,
      bundle.visual_patch_size_total);
  if (!reader.finished())
    throw std::runtime_error("trailing bytes in recovery handoff");
  if (bundle.generation == 0 || !std::isfinite(bundle.boundary_relative_time) ||
      bundle.boundary_relative_time < 0.0)
    throw std::runtime_error("invalid recovery handoff boundary metadata");
  return bundle;
}

void LIVMapper::writeRecoveryHandoffBundleAtomic(
    const std::string &path, const RecoveryHandoffBundle &bundle)
{
  if (path.empty()) throw std::invalid_argument("recovery handoff path is empty");
  const std::vector<std::uint8_t> payload =
      serializeRecoveryHandoffBundle(bundle);
  const std::string temporary = path + ".tmp." +
      std::to_string(static_cast<long long>(::getpid()));
  {
    std::ofstream stream(temporary.c_str(),
                         std::ios::out | std::ios::binary | std::ios::trunc);
    if (!stream.is_open())
      throw std::runtime_error("cannot open recovery handoff temporary file");
    stream.write(reinterpret_cast<const char *>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    stream.flush();
    if (!stream)
    {
      stream.close();
      std::remove(temporary.c_str());
      throw std::runtime_error("cannot write recovery handoff temporary file");
    }
  }
  if (std::rename(temporary.c_str(), path.c_str()) != 0)
  {
    std::remove(temporary.c_str());
    throw std::runtime_error("cannot atomically publish recovery handoff");
  }
}

LIVMapper::RecoveryHandoffHeader LIVMapper::readRecoveryHandoffHeader(
    const std::string &path)
{
  std::ifstream stream(path.c_str(), std::ios::in | std::ios::binary);
  if (!stream.is_open())
    throw std::runtime_error("cannot open recovery handoff bundle");
  constexpr std::size_t header_size = sizeof(std::uint64_t) +
      sizeof(std::uint32_t) + sizeof(std::uint64_t) + sizeof(double) +
      sizeof(std::uint64_t) + sizeof(int);
  std::array<std::uint8_t, header_size> bytes{};
  stream.read(reinterpret_cast<char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (stream.gcount() != static_cast<std::streamsize>(bytes.size()))
    throw std::runtime_error("truncated recovery handoff header");
  Reader reader(bytes.data(), bytes.size());
  if (reader.pod<std::uint64_t>() != kRecoveryHandoffMagic)
    throw std::runtime_error("invalid recovery handoff magic");
  if (reader.pod<std::uint32_t>() != kRecoveryHandoffSchemaVersion)
    throw std::runtime_error("unsupported recovery handoff schema");
  RecoveryHandoffHeader header;
  header.generation = reader.pod<std::uint64_t>();
  header.boundary_relative_time = reader.pod<double>();
  header.next_event_sequence =
      static_cast<std::size_t>(reader.pod<std::uint64_t>());
  header.visual_patch_size_total = reader.pod<int>();
  if (header.generation == 0 ||
      !std::isfinite(header.boundary_relative_time) ||
      header.boundary_relative_time < 0.0 ||
      header.visual_patch_size_total < 0)
    throw std::runtime_error("invalid recovery handoff boundary metadata");
  return header;
}

LIVMapper::RecoveryHandoffBundle LIVMapper::readRecoveryHandoffBundle(
    const std::string &path)
{
  const ReadOnlyFileMapping mapping(path);
  return deserializeRecoveryHandoffBundle(mapping.data(), mapping.size());
}

std::string LIVMapper::authenticatePrivateForkJournalFiles(
    const std::string &bundle_journal_path,
    const std::string &sealed_journal_path,
    const std::string &writable_spill_path)
{
  if (bundle_journal_path.empty())
    return "private_fork_journal_path_missing";
  if (sealed_journal_path.empty() ||
      bundle_journal_path != sealed_journal_path)
    return "private_fork_journal_path_mismatch";
  if (!writable_spill_path.empty() &&
      bundle_journal_path == writable_spill_path)
    return "private_fork_journal_aliases_writable_spill";
  struct stat status;
  if (::stat(bundle_journal_path.c_str(), &status) != 0)
    return "private_fork_journal_stat_failed";
  if (!S_ISREG(status.st_mode))
    return "private_fork_journal_not_regular_file";
  if ((status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0)
    return "private_fork_journal_writable";
  const int fd = ::open(bundle_journal_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return "private_fork_journal_open_readonly_failed";
  ::close(fd);
  return std::string();
}

void LIVMapper::maybeExportRecoveryHandoff(
    const double incoming_relative_time)
{
  if (recovery_handoff_export_directory.empty() || !lidar_map_inited ||
       last_completed_event_relative_time < 0.0)
    return;
  if (recovery_handoff_stop_export_after_consume &&
      !recovery_handoff_export_consumed)
  {
    const std::string marker = joinPath(
        recovery_handoff_export_directory, "handoff_consumed.marker");
    struct stat marker_status;
    if (::stat(marker.c_str(), &marker_status) == 0)
    {
      std::size_t pruned_files = 0;
      for (const std::string &path : recovery_handoff_export_files)
      {
        if (std::remove(path.c_str()) == 0 || errno == ENOENT)
          ++pruned_files;
        else
          ROS_WARN_STREAM("[ Handoff ] Could not prune consumed seed " << path);
      }
      recovery_handoff_export_files.clear();
      recovery_handoff_export_consumed = true;
      publishTransactionStatus(
          "handoff_export_stopped",
          "reason=consumer_import_completed;pruned_files=" +
              std::to_string(pruned_files),
          last_completed_event_relative_time);
      ROS_WARN_STREAM("[ Handoff ] Consumer imported a seed; stopped export "
                      "and pruned " << pruned_files << " published bundles");
    }
    else if (errno != ENOENT)
    {
      ROS_WARN_STREAM("[ Handoff ] Could not inspect consume marker " << marker);
    }
  }
  if (recovery_handoff_export_consumed) return;
  if (last_recovery_handoff_export_time >= 0.0 &&
      incoming_relative_time - last_recovery_handoff_export_time <
          rolling_checkpoint_period_sec)
    return;
  struct stat directory_status;
  if (::stat(recovery_handoff_export_directory.c_str(), &directory_status) != 0)
  {
    if (errno != ENOENT ||
        ::mkdir(recovery_handoff_export_directory.c_str(), 0755) != 0)
      throw std::runtime_error("cannot create recovery handoff directory");
  }
  else if (!S_ISDIR(directory_status.st_mode))
  {
    throw std::runtime_error("recovery handoff export path is not a directory");
  }

  const double capture_start = omp_get_wtime();
  RecoveryHandoffBundle bundle;
  bundle.generation = ++recovery_handoff_generation;
  bundle.boundary_relative_time = std::nextafter(
      last_completed_event_relative_time,
      -std::numeric_limits<double>::infinity());
  bundle.next_event_sequence = next_transaction_event_sequence;
  bundle.visual_patch_size_total = vio_manager->patch_size_total;
  bundle.mapper = captureMapperCheckpoint();
  bundle.voxel_map = voxelmap_manager->CreateSnapshot();
  bundle.visual_map = vio_manager->createCheckpoint();
  bundle.detector = harm_detector->createSnapshot();
  if (private_fork_bind_journal)
  {
    replay_payload_journal.commitDurable();
    if (replay_payload_journal.path().empty())
      throw std::runtime_error("private fork journal path is empty at export");
    bundle.replay_journal_path = replay_payload_journal.path();
  }
  const long long boundary_microseconds = static_cast<long long>(std::llround(
      bundle.boundary_relative_time * 1000000.0));
  std::ostringstream filename;
  filename << "handoff_g" << std::setw(10) << std::setfill('0')
           << bundle.generation << "_t" << std::setw(16)
           << boundary_microseconds << ".bin";
  const std::string path = joinPath(
      recovery_handoff_export_directory, filename.str());
  writeRecoveryHandoffBundleAtomic(path, bundle);
  recovery_handoff_export_files.push_back(path);
  while (recovery_handoff_export_files.size() > rolling_max_checkpoints)
  {
    const std::string expired = recovery_handoff_export_files.front();
    recovery_handoff_export_files.pop_front();
    bool keep_for_snapshot = false;
    if (typed_event_map_snapshot_enabled &&
        typed_event_map_snapshot_branch_role == "direct")
    {
      try
      {
        const double completed =
            fast_livo::typed_event_map::completedTimeFromBoundaryLabel(
                readRecoveryHandoffHeader(expired).boundary_relative_time);
        keep_for_snapshot = std::isfinite(completed) &&
            std::abs(completed -
                     typed_event_map_snapshot_checkpoint_completed) <= 1e-9;
      }
      catch (const std::exception &)
      {
        keep_for_snapshot = false;
      }
    }
    if (keep_for_snapshot)
    {
      recovery_handoff_export_files.push_back(expired);
      if (recovery_handoff_export_files.front() == expired) break;
      continue;
    }
    if (std::remove(expired.c_str()) != 0 && errno != ENOENT)
      ROS_WARN_STREAM("[ Handoff ] Could not prune " << expired);
  }
  last_recovery_handoff_export_time = incoming_relative_time;
  std::ostringstream detail;
  detail << std::setprecision(17)
         << "generation=" << bundle.generation
         << ";boundary_relative_time_sec=" << bundle.boundary_relative_time
         << ";export_wall_sec=" << (omp_get_wtime() - capture_start)
         << ";voxel_count=" << bundle.voxel_map->size()
         << ";visual_voxel_count=" << bundle.visual_map->feat_map.size()
         << ";path=" << path;
  publishTransactionStatus("handoff_exported", detail.str(),
                           bundle.boundary_relative_time);
}

void LIVMapper::markRecoveryHandoffConsumed(
    const std::uint64_t generation, const double boundary_relative_time)
{
  if (!recovery_handoff_stop_export_after_consume ||
      recovery_handoff_import_directory.empty())
    return;
  const std::string marker = joinPath(
      recovery_handoff_import_directory, "handoff_consumed.marker");
  const std::string temporary = marker + ".tmp." +
      std::to_string(static_cast<long long>(::getpid()));
  {
    std::ofstream stream(temporary.c_str(), std::ios::out | std::ios::trunc);
    if (!stream.is_open())
      throw std::runtime_error("cannot open recovery handoff consume marker");
    stream << std::setprecision(17)
           << "generation=" << generation << '\n'
           << "boundary_relative_time_sec=" << boundary_relative_time << '\n';
    stream.flush();
    if (!stream)
    {
      stream.close();
      std::remove(temporary.c_str());
      throw std::runtime_error("cannot write recovery handoff consume marker");
    }
  }
  if (std::rename(temporary.c_str(), marker.c_str()) != 0)
  {
    std::remove(temporary.c_str());
    throw std::runtime_error(
        "cannot atomically publish recovery handoff consume marker");
  }
}

std::unique_ptr<LIVMapper::RecoveryHandoffBundle>
LIVMapper::loadRecoveryHandoffBefore(const double onset_relative_time) const
{
  if (recovery_handoff_import_directory.empty()) return nullptr;
  DIR *directory = ::opendir(recovery_handoff_import_directory.c_str());
  if (directory == nullptr)
    throw std::runtime_error("cannot open recovery handoff import directory");
  std::vector<std::string> paths;
  while (dirent *entry = ::readdir(directory))
  {
    const std::string name(entry->d_name);
    if (name.size() < 4 || name.substr(name.size() - 4) != ".bin") continue;
    paths.push_back(joinPath(recovery_handoff_import_directory, name));
  }
  ::closedir(directory);
  struct IndexedHandoff
  {
    std::string path;
    RecoveryHandoffHeader header;
  };
  std::vector<IndexedHandoff> candidates;
  for (const std::string &path : paths)
  {
    try
    {
      RecoveryHandoffHeader header = readRecoveryHandoffHeader(path);
      if (header.boundary_relative_time >= onset_relative_time) continue;
      candidates.push_back({path, header});
    }
    catch (const std::exception &error)
    {
      ROS_WARN_STREAM("[ Handoff ] Ignoring invalid bundle header " << path
                      << ": " << error.what());
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const IndexedHandoff &left, const IndexedHandoff &right) {
    if (left.header.boundary_relative_time !=
        right.header.boundary_relative_time)
      return left.header.boundary_relative_time >
             right.header.boundary_relative_time;
    if (left.header.generation != right.header.generation)
      return left.header.generation > right.header.generation;
    return left.path > right.path;
  });
  for (const IndexedHandoff &indexed : candidates)
  {
    try
    {
      std::string receipt_hash;
      if (private_candidate_map_receipt_enabled ||
          typed_event_map_snapshot_enabled)
      {
        try { receipt_hash = fast_livo::source_revision::sha256File(indexed.path); }
        catch (const std::exception &error) { ROS_ERROR_STREAM("[ PrivateMapReceipt ] " << error.what()); }
      }
      RecoveryHandoffBundle selected = readRecoveryHandoffBundle(indexed.path);
      if (!receipt_hash.empty())
      {
        try
        {
          if (fast_livo::source_revision::sha256File(indexed.path) == receipt_hash)
            selected.receipt_file_sha256 = receipt_hash;
        }
        catch (const std::exception &error) { ROS_ERROR_STREAM("[ PrivateMapReceipt ] " << error.what()); }
      }
      if (selected.generation != indexed.header.generation ||
          selected.boundary_relative_time !=
              indexed.header.boundary_relative_time ||
          selected.next_event_sequence !=
              indexed.header.next_event_sequence ||
          selected.visual_patch_size_total !=
              indexed.header.visual_patch_size_total ||
          selected.boundary_relative_time >= onset_relative_time)
        throw std::runtime_error("recovery handoff header changed during load");
      return std::unique_ptr<RecoveryHandoffBundle>(
          new RecoveryHandoffBundle(std::move(selected)));
    }
    catch (const std::exception &error)
    {
      ROS_WARN_STREAM("[ Handoff ] Ignoring invalid selected bundle "
                      << indexed.path << ": " << error.what());
    }
  }
  return nullptr;
}

void LIVMapper::initializeTransactionDiagnostics()
{
  if (!transaction_diagnostics_path.empty())
  {
    transaction_diagnostics.open(transaction_diagnostics_path.c_str(), std::ios::out | std::ios::trunc);
    if (!transaction_diagnostics.is_open())
    {
      ROS_ERROR_STREAM("[ TransactionDiagnostics ] Cannot open " << transaction_diagnostics_path);
    }
    else
    {
      transaction_diagnostics
          << "event,relative_time,sensor,buffer_index,quarantined,"
          << "pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,"
          << "vel_x,vel_y,vel_z,bias_g_x,bias_g_y,bias_g_z,"
          << "bias_a_x,bias_a_y,bias_a_z,gravity_x,gravity_y,gravity_z,"
          << "cov_trace,residual,match_ratio,raw_residual,raw_match_ratio,"
          << "normalized_innovation,"
          << "selective_inlier_ratio,selective_gate_threshold_m,"
          << "robust_weight_mean,robust_weight_minimum,robust_weight_transition_m,"
          << "voxel_count,visual_voxel_count,trajectory_count,pcd_point_count,"
          << "lidar_frame_count,visual_frame_count,delta_pos_m,delta_rot_deg,"
          << "delta_vel_mps,delta_bias_g,delta_bias_a,delta_cov_fro\n";
      transaction_diagnostics.flush();
      ROS_INFO_STREAM("[ TransactionDiagnostics ] Writing " << transaction_diagnostics_path);
    }
  }
  if (!trajectory_journal_path.empty())
  {
    trajectory_journal.open(trajectory_journal_path.c_str(), std::ios::out | std::ios::trunc);
    if (!trajectory_journal.is_open())
      ROS_ERROR_STREAM("[ TrajectoryJournal ] Cannot open " << trajectory_journal_path);
    else
    {
      trajectory_journal << "# timestamp x y z qx qy qz qw replay\n";
      trajectory_journal.flush();
      ROS_INFO_STREAM("[ TrajectoryJournal ] Writing " << trajectory_journal_path);
    }

  }
  if (!harm_detector_diagnostics_path.empty())
  {
    harm_detector_diagnostics.open(
        harm_detector_diagnostics_path.c_str(), std::ios::out | std::ios::trunc);
    if (!harm_detector_diagnostics.is_open())
      ROS_ERROR_STREAM("[ HarmDetectorDiagnostics ] Cannot open "
                       << harm_detector_diagnostics_path);
    else
    {
      harm_detector_diagnostics
          << "relative_time_sec,raw_residual,raw_match_ratio,normalized_innovation,"
             "valid_observation,baseline_ready,weak_evidence,anomalous,triggered,"
             "residual_score,match_score,innovation_score,fused_score,"
             "residual_threshold,match_ratio_threshold,innovation_threshold\n";
      harm_detector_diagnostics.flush();
      ROS_INFO_STREAM("[ HarmDetectorDiagnostics ] Writing "
                      << harm_detector_diagnostics_path);
    }
  }
}

void LIVMapper::logTransactionDiagnostic(const std::string &event,
                                         const double relative_time,
                                         const StatesGroup *reference,
                                         const long long buffer_index,
                                         const int quarantined)
{
  if (!transaction_diagnostics.is_open()) return;
  const Eigen::Quaterniond q(_state.rot_end);
  double delta_pos = 0.0;
  double delta_rot_deg = 0.0;
  double delta_vel = 0.0;
  double delta_bias_g = 0.0;
  double delta_bias_a = 0.0;
  double delta_cov = 0.0;
  if (reference != nullptr)
  {
    delta_pos = (_state.pos_end - reference->pos_end).norm();
    const M3D rotation_delta = reference->rot_end.transpose() * _state.rot_end;
    delta_rot_deg = Log(rotation_delta).norm() * 180.0 / M_PI;
    delta_vel = (_state.vel_end - reference->vel_end).norm();
    delta_bias_g = (_state.bias_g - reference->bias_g).norm();
    delta_bias_a = (_state.bias_a - reference->bias_a).norm();
    delta_cov = (_state.cov - reference->cov).norm();
  }
  const char *sensor = LidarMeasures.lio_vio_flg == LIO ? "LIO" : "VIO";
  transaction_diagnostics << std::setprecision(15)
      << event << ',' << relative_time << ',' << sensor << ',' << buffer_index << ',' << quarantined << ','
      << _state.pos_end.x() << ',' << _state.pos_end.y() << ',' << _state.pos_end.z() << ','
      << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << ','
      << _state.vel_end.x() << ',' << _state.vel_end.y() << ',' << _state.vel_end.z() << ','
      << _state.bias_g.x() << ',' << _state.bias_g.y() << ',' << _state.bias_g.z() << ','
      << _state.bias_a.x() << ',' << _state.bias_a.y() << ',' << _state.bias_a.z() << ','
      << _state.gravity.x() << ',' << _state.gravity.y() << ',' << _state.gravity.z() << ','
      << _state.cov.trace() << ',' << voxelmap_manager->mean_abs_residual_ << ','
      << voxelmap_manager->effective_match_ratio_ << ','
      << voxelmap_manager->raw_mean_abs_residual_ << ',' << voxelmap_manager->raw_match_ratio_ << ','
      << detector_normalized_innovation << ','
      << voxelmap_manager->selective_inlier_ratio_ << ','
      << voxelmap_manager->selective_gate_threshold_m_ << ','
      << voxelmap_manager->robust_weight_mean_ << ','
      << voxelmap_manager->robust_weight_minimum_ << ','
      << voxelmap_manager->robust_weight_transition_m_ << ','
      << voxelmap_manager->voxel_map_.size() << ','
      << vio_manager->feat_map.size() << ',' << trajectory_poses.size() << ','
      << pcl_wait_save->points.size() << ',' << frame_num << ',' << vio_manager->frame_count << ','
      << delta_pos << ',' << delta_rot_deg << ',' << delta_vel << ','
      << delta_bias_g << ',' << delta_bias_a << ',' << delta_cov << '\n';
  transaction_diagnostics.flush();
}

void LIVMapper::beginMapTransaction(const double relative_time)
{
  if (!lidar_map_inited)
  {
    ROS_ERROR_STREAM("[ Transaction ] Cannot checkpoint before the initial LiDAR map exists");
    map_transaction_state->markRolledBack();
    return;
  }

  {
    std::ostringstream detail;
    detail << "checkpoint_mode="
           << (delta_voxel_journal_enabled ? "delta_before_image" : "direct")
           << ";checkpoint_voxel_count=" << voxelmap_manager->voxel_map_.size()
           << ";checkpoint_visual_voxel_count=" << vio_manager->feat_map.size();
    publishTransactionStatus("checkpoint_started", detail.str(), relative_time);
  }

  transaction_event_clone_total_wall_sec = 0.0;
  transaction_event_clone_maximum_wall_sec = 0.0;
  transaction_event_clone_count = 0;
  peak_transaction_buffer_events = 0;
  const double checkpoint_total_start = omp_get_wtime();
  const double mapper_copy_start = omp_get_wtime();
  map_transaction_checkpoint = captureMapperCheckpoint();
  selected_checkpoint_mapper_copy_wall_sec = omp_get_wtime() - mapper_copy_start;

  if (!online_harm_detector_enabled && healthy_tail_config.enabled &&
      healthy_residual_history.size() >= 20 && healthy_match_history.size() >= 20)
  {
    const double residual_median = robustMedian(healthy_residual_history);
    const double residual_sigma = robustMad(healthy_residual_history, residual_median);
    const double match_median = robustMedian(healthy_match_history);
    const double match_sigma = robustMad(healthy_match_history, match_median);
    healthy_residual_limit = residual_median +
        std::max(healthy_residual_mad_multiplier * residual_sigma,
                 healthy_residual_increase_floor);
    healthy_match_ratio_limit = std::max(
        0.0, match_median -
        std::max(healthy_match_mad_multiplier * match_sigma,
                 healthy_match_drop_floor));
    healthy_match_baseline_reference = match_median;
    healthy_validator_limits_ready = true;
    ROS_INFO_STREAM("[ HealthyTail ] Oracle baseline residual_limit="
                    << healthy_residual_limit << " match_limit="
                    << healthy_match_ratio_limit << " samples="
                    << healthy_residual_history.size());
  }

  const double voxel_copy_start = omp_get_wtime();
  transaction_checkpoint_voxel_count = voxelmap_manager->voxel_map_.size();
  transaction_uses_delta_voxel_journal = delta_voxel_journal_enabled;
  if (transaction_uses_delta_voxel_journal)
  {
    voxel_map_checkpoint.reset();
    transaction_voxel_journal_id =
        voxelmap_manager->BeginMutationJournal();
  }
  else
  {
    voxel_map_checkpoint = voxelmap_manager->CreateSnapshot();
  }
  selected_checkpoint_voxel_copy_wall_sec = omp_get_wtime() - voxel_copy_start;
  const double visual_copy_start = omp_get_wtime();
  transaction_uses_delta_visual_journal = delta_voxel_journal_enabled;
  if (transaction_uses_delta_visual_journal)
  {
    vio_map_checkpoint = vio_manager->createMetadataCheckpoint();
    transaction_visual_journal_id = vio_manager->beginMutationJournal();
  }
  else
  {
    vio_map_checkpoint = vio_manager->createCheckpoint();
  }
  selected_checkpoint_visual_copy_wall_sec = omp_get_wtime() - visual_copy_start;
  selected_checkpoint_total_copy_wall_sec = omp_get_wtime() - checkpoint_total_start;
  map_transaction_buffer.clear();
  resetReplayPayloadJournal();
  logTransactionDiagnostic("checkpoint_open", relative_time, &map_transaction_checkpoint.state);
  {
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "checkpoint_mode="
           << (transaction_uses_delta_voxel_journal ?
               "delta_before_image" : "direct")
           << ";checkpoint_mapper_copy_wall_sec=" << selected_checkpoint_mapper_copy_wall_sec
           << ";checkpoint_voxel_copy_wall_sec=" << selected_checkpoint_voxel_copy_wall_sec
           << ";checkpoint_visual_copy_wall_sec=" << selected_checkpoint_visual_copy_wall_sec
           << ";checkpoint_total_copy_wall_sec=" << selected_checkpoint_total_copy_wall_sec
           << ";checkpoint_voxel_count=" << transaction_checkpoint_voxel_count
           << ";checkpoint_visual_voxel_count=" << vio_manager->feat_map.size()
           << ";visual_restore_mode="
           << (transaction_uses_delta_visual_journal ?
               "delta_rollback" : "full_snapshot")
           << ";visual_journal_id=" << transaction_visual_journal_id;
    publishTransactionStatus("transaction_started", detail.str(), relative_time);
  }
  ROS_WARN_STREAM("[ Transaction ] Opened at t=" << relative_time << " s with "
                  << transaction_checkpoint_voxel_count << " verified voxels and "
                  << vio_manager->feat_map.size() << " visual voxels");
}

bool LIVMapper::bufferMapTransactionEvent(const double relative_time)
{
  if (!map_transaction_state->recordBufferedEvent())
  {
    rollbackMapTransaction("buffer capacity exceeded");
    return false;
  }

  BufferedTransactionEvent event;
  event.sequence = next_transaction_event_sequence++;
  event.relative_time = relative_time;
  const double clone_start = omp_get_wtime();
  event.measures = cloneMeasureGroup(LidarMeasures);
  spillReplayPayload(event);
  const double clone_wall_sec = omp_get_wtime() - clone_start;
  transaction_event_clone_total_wall_sec += clone_wall_sec;
  transaction_event_clone_maximum_wall_sec =
      std::max(transaction_event_clone_maximum_wall_sec, clone_wall_sec);
  ++transaction_event_clone_count;
  map_transaction_buffer.push_back(std::move(event));
  peak_transaction_buffer_events =
      std::max(peak_transaction_buffer_events, map_transaction_buffer.size());
  return true;
}

void LIVMapper::restoreMapTransactionCheckpoint()
{
  restoreMapperCheckpoint(map_transaction_checkpoint);
  if (vio_map_checkpoint)
  {
    if (transaction_uses_delta_visual_journal)
      vio_manager->restoreCheckpointMetadata(*vio_map_checkpoint);
    else
      vio_manager->restoreCheckpoint(*vio_map_checkpoint);
  }
}

void LIVMapper::restoreMapperCheckpoint(const MapperCheckpoint &checkpoint)
{
  _state = checkpoint.state;
  state_propagat = checkpoint.propagated_state;
  latest_ekf_state = checkpoint.latest_ekf_state;
  imu_propagate = checkpoint.imu_propagate;
  p_imu->restoreCheckpoint(checkpoint.imu_process);
  _pv_list = checkpoint.point_variances;
  *pcl_w_wait_pub = checkpoint.lidar_world_for_visual;
  *pcl_wait_pub = checkpoint.pending_publish_cloud;
  voxelmap_manager->position_last_ = checkpoint.voxel_position_last;
  voxelmap_manager->last_slide_position = checkpoint.voxel_last_slide_position;
  latest_ekf_time = checkpoint.latest_ekf_time;
  last_completed_event_relative_time = checkpoint.last_completed_event_relative_time;
  aver_time_consu = checkpoint.average_processing_time;
  if (path.poses.size() > checkpoint.path_pose_count)
  {
    path.poses.resize(checkpoint.path_pose_count);
  }
  while (trajectory_poses.size() > checkpoint.trajectory_pose_count)
  {
    trajectory_poses.erase(std::prev(trajectory_poses.end()));
  }
  if (pcl_wait_save->points.size() > checkpoint.pcd_rgb_point_count)
  {
    pcl_wait_save->points.resize(checkpoint.pcd_rgb_point_count);
    pcl_wait_save->width = static_cast<uint32_t>(pcl_wait_save->points.size());
    pcl_wait_save->height = 1;
  }
  if (pcl_wait_save_intensity->points.size() > checkpoint.pcd_intensity_point_count)
  {
    pcl_wait_save_intensity->points.resize(checkpoint.pcd_intensity_point_count);
    pcl_wait_save_intensity->width = static_cast<uint32_t>(pcl_wait_save_intensity->points.size());
    pcl_wait_save_intensity->height = 1;
  }
  voxelmap_manager->current_frame_id_ = checkpoint.voxel_frame_id;
  if (checkpoint.voxel_next_plane_id >= 0)
    voxelmap_manager->restoreNextPlaneId(checkpoint.voxel_next_plane_id);
  Frame::frame_counter_ = checkpoint.visual_frame_counter;
  vio_manager->frame_count = checkpoint.visual_processed_frames;
  frame_num = checkpoint.processed_lidar_frames;
  lidar_map_inited = checkpoint.lidar_map_inited;
  ekf_finish_once = checkpoint.ekf_finish_once;
  state_update_flg = checkpoint.state_update_flg;
}

void LIVMapper::maintainRollingHistory(const double relative_time)
{
  if (!lidar_map_inited) return;
  const std::uint64_t history_profile_sequence =
      ++rolling_history_profile_sequence;
  const std::uint64_t history_stamp_ns = static_cast<std::uint64_t>(
      (_first_lidar_time + relative_time) * 1e9);
  fast_livo::audit::InputEventProfile::Scope history_profile_scope(
      input_event_profile, "rolling_history", history_profile_sequence, 0,
      history_stamp_ns, 0, relative_time);

  // A process-isolated candidate seeded from the public incumbent needs only
  // a time-complete sensor journal.  Maintaining candidate-local map
  // checkpoints and mutation before-images is both redundant and dangerous:
  // their rotation can prune events that are still required by the newest
  // fully-written incumbent handoff.  Retain an event-only horizon instead.
  if (!recovery_handoff_import_directory.empty())
  {
    if (rolling_event_buffer.empty() && disk_backed_replay_payload_enabled &&
        replay_payload_journal.recordsWritten() == 0)
      resetReplayPayloadJournal();
    BufferedTransactionEvent event;
    event.sequence = next_transaction_event_sequence++;
    event.relative_time = relative_time;
    const double clone_start = omp_get_wtime();
    {
      fast_livo::audit::InputEventProfile::Scope clone_profile_scope(
          input_event_profile, "rolling_clone", history_profile_sequence, 0,
          history_stamp_ns, 0, relative_time);
      event.measures = cloneMeasureGroup(LidarMeasures);
      clone_profile_scope.disposition("completed");
    }
    {
      fast_livo::audit::InputEventProfile::Scope spill_profile_scope(
          input_event_profile, "rolling_spill", history_profile_sequence, 0,
          history_stamp_ns, 0, relative_time);
      spillReplayPayload(event);
      spill_profile_scope.disposition("completed");
    }
    const double clone_wall_sec = omp_get_wtime() - clone_start;
    rolling_event_clone_total_wall_sec += clone_wall_sec;
    rolling_event_clone_maximum_wall_sec =
        std::max(rolling_event_clone_maximum_wall_sec, clone_wall_sec);
    ++rolling_event_clone_count;
    rolling_event_buffer.push_back(std::move(event));
    peak_rolling_event_buffer_events =
        std::max(peak_rolling_event_buffer_events,
                 rolling_event_buffer.size());
    fast_livo::audit::InputEventProfile::Scope rotate_profile_scope(
        input_event_profile, "rolling_rotate", history_profile_sequence, 0,
        history_stamp_ns, 0, relative_time);
    const double oldest_allowed_time =
        relative_time - rolling_history_horizon_sec;
    while (!rolling_event_buffer.empty() &&
           rolling_event_buffer.front().relative_time < oldest_allowed_time)
      rolling_event_buffer.pop_front();
    rotate_profile_scope.disposition("completed");
    history_profile_scope.disposition("event_only");
    return;
  }

  bool open_checkpoint = rolling_checkpoints.empty();
  if (!open_checkpoint &&
      relative_time - rolling_checkpoints.back().relative_time >=
          rolling_checkpoint_period_sec)
  {
    if (delta_voxel_journal_enabled)
    {
      // Keep one overlapping predecessor generation. The detector's onset
      // lower bound may precede the first strong/weak sample, so replacing the
      // only journal at this point would create a short unprotected interval.
      if (rolling_checkpoint_rotation_safe)
        open_checkpoint = true;
    }
    else
    {
      open_checkpoint = true;
    }
  }

  if (open_checkpoint)
  {
    if (rolling_checkpoints.empty() && rolling_event_buffer.empty() &&
        disk_backed_replay_payload_enabled &&
        replay_payload_journal.recordsWritten() == 0)
      resetReplayPayloadJournal();
    RollingCheckpoint checkpoint;
    // maintainRollingHistory() runs after sync_packages() has exposed the next
    // event but before that event is processed. Label the snapshot with the
    // last event whose mapping update actually completed; latest_ekf_time has
    // configuration-dependent absolute/relative semantics and is not a safe
    // transaction boundary clock.
    const double checkpoint_state_time = last_completed_event_relative_time;
    checkpoint.relative_time = std::nextafter(
        checkpoint_state_time, -std::numeric_limits<double>::infinity());
    checkpoint.first_event_sequence = next_transaction_event_sequence;
    checkpoint.voxel_count = voxelmap_manager->voxel_map_.size();
    checkpoint.uses_delta_voxel_journal = delta_voxel_journal_enabled;
    const double checkpoint_total_start = omp_get_wtime();
    const double mapper_copy_start = omp_get_wtime();
    checkpoint.mapper = captureMapperCheckpoint();
    checkpoint.mapper_copy_wall_sec = omp_get_wtime() - mapper_copy_start;
    const double voxel_copy_start = omp_get_wtime();
    if (checkpoint.uses_delta_voxel_journal)
    {
      checkpoint.voxel_journal_id =
          voxelmap_manager->BeginMutationJournal();
    }
    else
    {
      checkpoint.voxel_map = voxelmap_manager->CreateSnapshot();
    }
    checkpoint.voxel_copy_wall_sec = omp_get_wtime() - voxel_copy_start;
    const double visual_copy_start = omp_get_wtime();
    checkpoint.uses_delta_visual_journal = delta_voxel_journal_enabled;
    if (checkpoint.uses_delta_visual_journal)
    {
      checkpoint.visual_map = vio_manager->createMetadataCheckpoint();
      checkpoint.visual_journal_id = vio_manager->beginMutationJournal();
    }
    else
    {
      checkpoint.visual_map = vio_manager->createCheckpoint();
    }
    checkpoint.visual_copy_wall_sec = omp_get_wtime() - visual_copy_start;
    checkpoint.total_copy_wall_sec = omp_get_wtime() - checkpoint_total_start;
    {
      std::ostringstream detail;
      detail << std::setprecision(17)
             << "checkpoint_mode="
             << (checkpoint.uses_delta_voxel_journal ?
                 "rolling_delta_anchor" : "rolling")
             << ";checkpoint_mapper_copy_wall_sec=" << checkpoint.mapper_copy_wall_sec
             << ";checkpoint_voxel_copy_wall_sec=" << checkpoint.voxel_copy_wall_sec
             << ";checkpoint_visual_copy_wall_sec=" << checkpoint.visual_copy_wall_sec
             << ";checkpoint_total_copy_wall_sec=" << checkpoint.total_copy_wall_sec
             << ";checkpoint_relative_time_sec=" << checkpoint.relative_time
             << ";voxel_journal_id=" << checkpoint.voxel_journal_id
             << ";visual_journal_id=" << checkpoint.visual_journal_id
             << ";checkpoint_voxel_count=" << checkpoint.voxel_count
             << ";checkpoint_visual_voxel_count=" << vio_manager->feat_map.size()
             << ";rolling_checkpoint_count=" << (rolling_checkpoints.size() + 1);
      publishTransactionStatus(
          "rolling_checkpoint_profile", detail.str(), checkpoint.relative_time);
    }
    ROS_INFO_STREAM("[ RollingHistory ] checkpoint state t="
                    << checkpoint.relative_time << " before incoming event t="
                    << relative_time
                    << " geometric_voxels=" << checkpoint.voxel_count
                    << " mode="
                    << (checkpoint.uses_delta_voxel_journal ?
                        "delta_anchor" : "full_snapshot")
                    << " visual_voxels=" << vio_manager->feat_map.size());
    rolling_checkpoints.push_back(std::move(checkpoint));
  }

  BufferedTransactionEvent event;
  event.sequence = next_transaction_event_sequence++;
  event.relative_time = relative_time;
  const double clone_start = omp_get_wtime();
  {
    fast_livo::audit::InputEventProfile::Scope clone_profile_scope(
        input_event_profile, "rolling_clone", history_profile_sequence, 0,
        history_stamp_ns, 0, relative_time);
    event.measures = cloneMeasureGroup(LidarMeasures);
    clone_profile_scope.disposition("completed");
  }
  {
    fast_livo::audit::InputEventProfile::Scope spill_profile_scope(
        input_event_profile, "rolling_spill", history_profile_sequence, 0,
        history_stamp_ns, 0, relative_time);
    spillReplayPayload(event);
    spill_profile_scope.disposition("completed");
  }
  const double clone_wall_sec = omp_get_wtime() - clone_start;
  rolling_event_clone_total_wall_sec += clone_wall_sec;
  rolling_event_clone_maximum_wall_sec =
      std::max(rolling_event_clone_maximum_wall_sec, clone_wall_sec);
  ++rolling_event_clone_count;
  rolling_event_buffer.push_back(std::move(event));
  peak_rolling_event_buffer_events =
      std::max(peak_rolling_event_buffer_events, rolling_event_buffer.size());

  fast_livo::audit::InputEventProfile::Scope rotate_profile_scope(
      input_event_profile, "rolling_rotate", history_profile_sequence, 0,
      history_stamp_ns, 0, relative_time);
  const double cutoff = relative_time - rolling_history_horizon_sec;
  const auto protected_front = [this]()
  {
    return !rolling_checkpoints.empty() &&
        protected_rolling_checkpoint_time_sec >= 0.0 &&
        std::abs(rolling_checkpoints.front().relative_time -
                 protected_rolling_checkpoint_time_sec) <= 1e-9;
  };
  while (rolling_checkpoints.size() > 1 &&
         rolling_checkpoints[1].relative_time <= cutoff)
  {
    if (protected_front())
    {
      auto discard = std::next(rolling_checkpoints.begin());
      if (discard->uses_delta_voxel_journal)
        voxelmap_manager->DiscardMutationJournal(discard->voxel_journal_id);
      if (discard->uses_delta_visual_journal)
        vio_manager->discardMutationJournal(discard->visual_journal_id);
      rolling_checkpoints.erase(discard);
    }
    else
    {
      if (rolling_checkpoints.front().uses_delta_voxel_journal)
        voxelmap_manager->DiscardMutationJournal(
            rolling_checkpoints.front().voxel_journal_id);
      if (rolling_checkpoints.front().uses_delta_visual_journal)
        vio_manager->discardMutationJournal(
            rolling_checkpoints.front().visual_journal_id);
      rolling_checkpoints.pop_front();
    }
  }
  const std::size_t retained_checkpoint_limit = delta_voxel_journal_enabled ?
      std::min(rolling_max_checkpoints, rolling_delta_max_checkpoints) :
      rolling_max_checkpoints;
  while (rolling_checkpoints.size() > retained_checkpoint_limit)
  {
    const bool preserve_front_for_horizon =
        delta_voxel_journal_enabled && rolling_preserve_horizon_anchor &&
        rolling_checkpoints.size() > 1;
    if (protected_front() || preserve_front_for_horizon)
    {
      auto discard = std::next(rolling_checkpoints.begin());
      if (discard->uses_delta_voxel_journal)
        voxelmap_manager->DiscardMutationJournal(discard->voxel_journal_id);
      if (discard->uses_delta_visual_journal)
        vio_manager->discardMutationJournal(discard->visual_journal_id);
      rolling_checkpoints.erase(discard);
    }
    else
    {
      if (rolling_checkpoints.front().uses_delta_voxel_journal)
        voxelmap_manager->DiscardMutationJournal(
            rolling_checkpoints.front().voxel_journal_id);
      if (rolling_checkpoints.front().uses_delta_visual_journal)
        vio_manager->discardMutationJournal(
            rolling_checkpoints.front().visual_journal_id);
      rolling_checkpoints.pop_front();
    }
  }
  if (!rolling_checkpoints.empty())
  {
    const std::size_t first_sequence = rolling_checkpoints.front().first_event_sequence;
    while (!rolling_event_buffer.empty() && rolling_event_buffer.front().sequence < first_sequence)
    {
      rolling_event_buffer.pop_front();
    }
  }
  rotate_profile_scope.disposition("completed");
  history_profile_scope.disposition("completed");
}

bool LIVMapper::activateRollingTransaction(const double relative_time)
{
  private_candidate_checkpoint_sha256.clear();
  private_candidate_handoff_generation = 0;
  const double estimated_onset = map_transaction_state->quarantineStartTime();
  std::unique_ptr<RecoveryHandoffBundle> imported;
  RecoveryReplayCoverage imported_coverage;
  std::string handoff_failure_reason;
  if (!recovery_handoff_import_directory.empty())
  {
    try
    {
      imported = loadRecoveryHandoffBefore(estimated_onset);
    }
    catch (const std::exception &error)
    {
      handoff_failure_reason = error.what();
      ROS_ERROR_STREAM("[ Handoff ] Import scan failed: " << error.what());
      publishTransactionStatus("handoff_import_failed", error.what(), relative_time);
    }
  }
  if (imported != nullptr)
  {
    if (imported->visual_patch_size_total != vio_manager->patch_size_total)
    {
      handoff_failure_reason = "visual_patch_size_mismatch";
      publishTransactionStatus(
          "handoff_import_failed", handoff_failure_reason, relative_time);
      imported.reset();
    }
  }
  if (imported != nullptr)
  {
    // The serialized boundary label is one ULP before the last completed
    // event, while the mapper checkpoint already contains that event's state.
    // Coverage must start strictly after the completed state time.
    imported_coverage = assessRecoveryReplayCoverage(
        imported->mapper.last_completed_event_relative_time,
        rolling_event_buffer,
        recovery_handoff_maximum_replay_gap_sec);
    if (imported_coverage.continuous && private_fork_require_readonly_journal)
    {
      const std::string journal_reason = authenticatePrivateForkJournalFiles(
          imported->replay_journal_path, private_fork_sealed_journal_path,
          replay_payload_journal_path);
      if (!journal_reason.empty())
      {
        imported_coverage.continuous = false;
        handoff_failure_reason = journal_reason;
        ROS_ERROR_STREAM("[ Handoff ] Refusing private fork journal: "
                         << journal_reason);
        publishTransactionStatus(
            "handoff_import_failed", journal_reason, relative_time);
        imported.reset();
      }
    }
    if (imported != nullptr && !imported_coverage.continuous)
    {
      std::ostringstream detail;
      detail << std::setprecision(17)
             << "reason=replay_history_gap"
             << ";checkpoint_relative_time_sec="
             << imported->boundary_relative_time
             << ";first_replay_event_relative_time_sec="
             << imported_coverage.first_event_relative_time
             << ";replay_gap_sec=" << imported_coverage.gap_sec
             << ";maximum_replay_gap_sec="
             << recovery_handoff_maximum_replay_gap_sec
             << ";rolling_event_count=" << rolling_event_buffer.size();
      handoff_failure_reason = detail.str();
      ROS_ERROR_STREAM("[ Handoff ] Refusing discontinuous replay: "
                       << handoff_failure_reason);
      publishTransactionStatus(
          "handoff_import_failed", handoff_failure_reason, relative_time);
      imported.reset();
    }
  }
  if (!recovery_handoff_import_directory.empty() && imported == nullptr)
  {
    if (handoff_failure_reason.empty())
      handoff_failure_reason = "no_eligible_exact_handoff";
    publishTransactionStatus(
        "candidate_rejected",
        "exact_handoff_required;" + handoff_failure_reason, relative_time);
    map_transaction_state->markRolledBack();
    if (voxelmap_manager->MutationJournalActive())
      voxelmap_manager->CommitMutationJournal();
    if (vio_manager->mutationJournalActive())
      vio_manager->commitMutationJournals();
    rolling_checkpoints.clear();
    rolling_event_buffer.clear();
    protected_rolling_checkpoint_time_sec = -1.0;
    if (!shadow_abort_topic.empty()) ros::shutdown();
    return false;
  }
  if (imported != nullptr)
  {
    try
    {
      markRecoveryHandoffConsumed(
          imported->generation, imported->boundary_relative_time);
    }
    catch (const std::exception &error)
    {
      ROS_WARN_STREAM("[ Handoff ] Could not publish consume marker: "
                      << error.what());
      publishTransactionStatus(
          "handoff_consume_marker_failed", error.what(), relative_time);
    }
  }
  if (imported != nullptr)
  {
    const double selected_time = imported->boundary_relative_time;
    const double completed_state_time =
        imported->mapper.last_completed_event_relative_time;
    private_candidate_checkpoint_sha256 = imported->receipt_file_sha256;
    private_candidate_handoff_generation = imported->generation;
    selected_checkpoint_relative_time_sec = selected_time;
    // Candidate-local journals describe a separately integrated generation;
    // discard their rollback metadata before installing the public seed.
    if (voxelmap_manager->MutationJournalActive())
      voxelmap_manager->CommitMutationJournal();
    if (vio_manager->mutationJournalActive())
      vio_manager->commitMutationJournals();
    transaction_uses_delta_voxel_journal = false;
    transaction_voxel_journal_id = 0;
    transaction_uses_delta_visual_journal = false;
    transaction_visual_journal_id = 0;
    map_transaction_checkpoint = imported->mapper;
    voxel_map_checkpoint = std::move(imported->voxel_map);
    vio_map_checkpoint = imported->visual_map;
    harm_detector->restoreSnapshot(imported->detector);
    selected_checkpoint_mapper_copy_wall_sec = 0.0;
    selected_checkpoint_voxel_copy_wall_sec = 0.0;
    selected_checkpoint_visual_copy_wall_sec = 0.0;
    selected_checkpoint_total_copy_wall_sec = 0.0;
    transaction_checkpoint_voxel_count = voxel_map_checkpoint->size();
    map_transaction_buffer.clear();
    transaction_event_clone_total_wall_sec = rolling_event_clone_total_wall_sec;
    transaction_event_clone_maximum_wall_sec = rolling_event_clone_maximum_wall_sec;
    transaction_event_clone_count = rolling_event_clone_count;
    peak_transaction_buffer_events = peak_rolling_event_buffer_events;
    for (BufferedTransactionEvent &event : rolling_event_buffer)
    {
      if (!fast_livo::replay::isAfterCompletedCheckpointEvent(
              event.relative_time, completed_state_time)) continue;
      if (!map_transaction_state->recordBufferedEvent())
      {
        rollbackMapTransaction("imported handoff journal exceeds transaction capacity");
        rolling_checkpoints.clear();
        rolling_event_buffer.clear();
        return false;
      }
      map_transaction_buffer.push_back(std::move(event));
    }
    rolling_checkpoints.clear();
    rolling_event_buffer.clear();
    protected_rolling_checkpoint_time_sec = -1.0;
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "checkpoint_mode=cross_process_handoff"
           << ";handoff_generation=" << imported->generation
           << ";checkpoint_relative_time_sec=" << selected_time
           << ";first_replay_event_relative_time_sec="
           << imported_coverage.first_event_relative_time
           << ";replay_gap_sec=" << imported_coverage.gap_sec
           << ";maximum_replay_gap_sec="
           << recovery_handoff_maximum_replay_gap_sec
           << ";checkpoint_voxel_count=" << transaction_checkpoint_voxel_count
           << ";checkpoint_visual_voxel_count=" << vio_map_checkpoint->feat_map.size()
           << ";transaction_buffer_events=" << map_transaction_buffer.size()
           << ";event_clone_count=" << transaction_event_clone_count
           << ";event_clone_total_wall_sec=" << transaction_event_clone_total_wall_sec
           << ";event_clone_maximum_wall_sec=" << transaction_event_clone_maximum_wall_sec
           << ";peak_transaction_buffer_events=" << peak_transaction_buffer_events;
    publishTransactionStatus("transaction_started", detail.str(), relative_time);
    ROS_WARN_STREAM("[ Handoff ] detection t=" << relative_time
                    << " imported incumbent generation " << imported->generation
                    << " at t=" << selected_time
                    << " before estimated onset t=" << estimated_onset
                    << "; promoted " << map_transaction_buffer.size()
                    << " candidate-local historical events");
    return true;
  }

  auto selected = rolling_checkpoints.end();
  for (auto it = rolling_checkpoints.begin(); it != rolling_checkpoints.end(); ++it)
  {
    if (it->relative_time < estimated_onset) selected = it;
  }
  if (selected == rolling_checkpoints.end())
  {
    ROS_WARN_STREAM("[ RollingHistory ] proposal deferred because no retained "
                    "checkpoint precedes estimated onset t=" << estimated_onset);
    if (shadow_rearm_after_internal_rejection)
    {
      // No replay or candidate generation has started, so terminating the
      // isolated mapper here would make an early natural alert mask every
      // later recoverable fault. Preserve the rolling journals, consume this
      // episode's attempt budget, and continue collecting verified anchors.
      const bool attempt_budget_refunded =
          recovery_proposal_arbiter_config.enabled &&
          recovery_proposal_arbiter->refundLastAttempt();
      publishTransactionStatus(
          "recovery_deferred_no_anchor",
          "reason=no_retained_checkpoint_before_onset;candidate_created=0;"
          "expensive_replay_started=0;monitoring_rearmed=1;"
          "attempt_budget_refunded=" +
              std::to_string(static_cast<int>(attempt_budget_refunded)),
          relative_time);
      map_transaction_state->markRolledBack();
      map_transaction_state->resetForMonitoring();
      harm_detector->resetForMonitoring();
      online_detection_pending = false;
      online_point_onset_sec = -1.0;
      online_detected_onset_sec = -1.0;
      online_detection_time_sec = -1.0;
      healthy_validator_limits_ready = false;
      healthy_residual_limit = 0.0;
      healthy_match_ratio_limit = 0.0;
      healthy_match_baseline_reference = 0.0;
      protected_rolling_checkpoint_time_sec = -1.0;
      rolling_checkpoint_rotation_safe = true;
      lidar_map_writes_enabled = true;
      visual_map_writes_enabled = true;
      return false;
    }
    publishTransactionStatus(
        "candidate_rejected", "no_retained_checkpoint_before_onset",
        relative_time);
    // A process-isolated shadow has no useful fallback branch of its own.
    // Exit at the detection site instead of waiting for the single-threaded
    // mapper loop to service the mux's asynchronous abort callback.
    if (!shadow_abort_topic.empty()) ros::shutdown();
    map_transaction_state->markRolledBack();
    if (voxelmap_manager->MutationJournalActive())
      voxelmap_manager->CommitMutationJournal();
    if (vio_manager->mutationJournalActive())
      vio_manager->commitMutationJournals();
    rolling_checkpoints.clear();
    rolling_event_buffer.clear();
    protected_rolling_checkpoint_time_sec = -1.0;
    return false;
  }

  const double selected_time = selected->relative_time;
  selected_checkpoint_relative_time_sec = selected_time;
  const std::size_t first_sequence = selected->first_event_sequence;
  transaction_uses_delta_voxel_journal =
      selected->uses_delta_voxel_journal;
  transaction_voxel_journal_id = selected->voxel_journal_id;
  transaction_uses_delta_visual_journal =
      selected->uses_delta_visual_journal;
  transaction_visual_journal_id = selected->visual_journal_id;
  if (transaction_uses_delta_voxel_journal &&
      !voxelmap_manager->MutationJournalActive(transaction_voxel_journal_id))
  {
    ROS_ERROR_STREAM("[ RollingHistory ] selected delta anchor has no active journal");
    publishTransactionStatus(
        "candidate_rejected", "rolling_delta_anchor_missing_journal",
        relative_time);
    map_transaction_state->markRolledBack();
    if (voxelmap_manager->MutationJournalActive())
      voxelmap_manager->CommitMutationJournal();
    if (vio_manager->mutationJournalActive())
      vio_manager->commitMutationJournals();
    rolling_checkpoints.clear();
    rolling_event_buffer.clear();
    protected_rolling_checkpoint_time_sec = -1.0;
    if (!shadow_abort_topic.empty()) ros::shutdown();
    return false;
  }
  if (transaction_uses_delta_visual_journal &&
      !vio_manager->mutationJournalActive(transaction_visual_journal_id))
  {
    ROS_ERROR_STREAM("[ RollingHistory ] selected visual delta anchor has no active journal");
    publishTransactionStatus(
        "candidate_rejected", "rolling_visual_delta_anchor_missing_journal",
        relative_time);
    map_transaction_state->markRolledBack();
    if (voxelmap_manager->MutationJournalActive())
      voxelmap_manager->CommitMutationJournal();
    if (vio_manager->mutationJournalActive())
      vio_manager->commitMutationJournals();
    rolling_checkpoints.clear();
    rolling_event_buffer.clear();
    protected_rolling_checkpoint_time_sec = -1.0;
    if (!shadow_abort_topic.empty()) ros::shutdown();
    return false;
  }
  // Once the detector has selected an anchor, overlapping successors can no
  // longer participate in recovery. Retire them immediately so the selected
  // journal can reuse their heap during quarantine instead of carrying two
  // equivalent before-image sets all the way to replay.
  if (transaction_uses_delta_voxel_journal)
  {
    for (const RollingCheckpoint &checkpoint : rolling_checkpoints)
    {
      if (checkpoint.uses_delta_voxel_journal &&
          checkpoint.voxel_journal_id != transaction_voxel_journal_id)
        voxelmap_manager->DiscardMutationJournal(
            checkpoint.voxel_journal_id);
    }
  }
  if (transaction_uses_delta_visual_journal)
  {
    for (const RollingCheckpoint &checkpoint : rolling_checkpoints)
    {
      if (checkpoint.uses_delta_visual_journal &&
          checkpoint.visual_journal_id != transaction_visual_journal_id)
        vio_manager->discardMutationJournal(checkpoint.visual_journal_id);
    }
  }
  map_transaction_checkpoint = selected->mapper;
  selected_checkpoint_mapper_copy_wall_sec = selected->mapper_copy_wall_sec;
  selected_checkpoint_voxel_copy_wall_sec = selected->voxel_copy_wall_sec;
  selected_checkpoint_visual_copy_wall_sec = selected->visual_copy_wall_sec;
  selected_checkpoint_total_copy_wall_sec = selected->total_copy_wall_sec;
  transaction_checkpoint_voxel_count = selected->voxel_count;
  if (transaction_uses_delta_voxel_journal)
    voxel_map_checkpoint.reset();
  else
    voxel_map_checkpoint = std::move(selected->voxel_map);
  vio_map_checkpoint = selected->visual_map;
  map_transaction_buffer.clear();
  transaction_event_clone_total_wall_sec = rolling_event_clone_total_wall_sec;
  transaction_event_clone_maximum_wall_sec = rolling_event_clone_maximum_wall_sec;
  transaction_event_clone_count = rolling_event_clone_count;
  peak_transaction_buffer_events = peak_rolling_event_buffer_events;

  for (BufferedTransactionEvent &event : rolling_event_buffer)
  {
    if (event.sequence < first_sequence) continue;
    if (!map_transaction_state->recordBufferedEvent())
    {
      rollbackMapTransaction("rolling journal exceeds transaction capacity");
      rolling_checkpoints.clear();
      rolling_event_buffer.clear();
      return false;
    }
    map_transaction_buffer.push_back(std::move(event));
  }
  rolling_checkpoints.clear();
  rolling_event_buffer.clear();
  protected_rolling_checkpoint_time_sec = -1.0;
  {
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "checkpoint_mode="
           << (transaction_uses_delta_voxel_journal ?
               "rolling_delta_selected" : "rolling_selected")
           << ";checkpoint_relative_time_sec=" << selected_time
           << ";voxel_journal_id=" << transaction_voxel_journal_id
           << ";visual_journal_id=" << transaction_visual_journal_id
           << ";visual_restore_mode="
           << (transaction_uses_delta_visual_journal ?
               "delta_rollback" : "full_snapshot")
           << ";checkpoint_mapper_copy_wall_sec=" << selected_checkpoint_mapper_copy_wall_sec
           << ";checkpoint_voxel_copy_wall_sec=" << selected_checkpoint_voxel_copy_wall_sec
           << ";checkpoint_visual_copy_wall_sec=" << selected_checkpoint_visual_copy_wall_sec
           << ";checkpoint_total_copy_wall_sec=" << selected_checkpoint_total_copy_wall_sec
           << ";transaction_buffer_events=" << map_transaction_buffer.size()
           << ";event_clone_count=" << transaction_event_clone_count
           << ";event_clone_total_wall_sec=" << transaction_event_clone_total_wall_sec
           << ";event_clone_maximum_wall_sec=" << transaction_event_clone_maximum_wall_sec
           << ";peak_transaction_buffer_events=" << peak_transaction_buffer_events;
    publishTransactionStatus("transaction_started", detail.str(), relative_time);
  }
  ROS_WARN_STREAM("[ RollingHistory ] detection t=" << relative_time
                  << " selected checkpoint t=" << selected_time
                  << " before estimated onset t=" << estimated_onset
                  << "; promoted " << map_transaction_buffer.size() << " historical events");
  return true;
}

void LIVMapper::rollbackMapTransaction(const std::string &reason)
{
  ROS_ERROR_STREAM("[ Transaction ] Rolling back: " << reason);
  publishTransactionStatus("rolled_back", reason);
  if (transaction_uses_delta_voxel_journal &&
      voxelmap_manager->MutationJournalActive(transaction_voxel_journal_id))
    voxelmap_manager->RollbackMutationJournal(transaction_voxel_journal_id);
  else if (voxel_map_checkpoint)
    voxelmap_manager->RestoreSnapshot(*voxel_map_checkpoint);
  if (transaction_uses_delta_visual_journal &&
      vio_manager->mutationJournalActive(transaction_visual_journal_id))
    vio_manager->rollbackMutationJournal(transaction_visual_journal_id);

  restoreMapTransactionCheckpoint();

  const auto phase = map_transaction_state->phase();
  if (map_transaction_state->active() || phase == fast_livo::transactional_mapping::Phase::OVERFLOWED)
  {
    map_transaction_state->markRolledBack();
  }
  voxel_map_checkpoint.reset();
  transaction_uses_delta_voxel_journal = false;
  transaction_checkpoint_voxel_count = 0;
  transaction_voxel_journal_id = 0;
  transaction_uses_delta_visual_journal = false;
  transaction_visual_journal_id = 0;
  vio_map_checkpoint.reset();
  map_transaction_buffer.clear();
  replay_incumbent_boundary.reset();
  replay_incumbent_state_valid = false;
  transaction_replay_active = false;
  lidar_updates_enabled = true;
  visual_updates_enabled = true;
  lidar_map_writes_enabled = true;
  visual_map_writes_enabled = true;
  if (shadow_rearm_after_internal_rejection)
  {
    publishTransactionStatus("candidate_terminated",
                             "unrecoverable_internal_rollback");
    ros::shutdown();
  }
}

void LIVMapper::rejectCandidateAndRestoreIncumbent(const std::string &reason)
{
  typed_event_snapshot_closed = true;
  ROS_ERROR_STREAM("[ Transaction ] Candidate rejected: " << reason);
  if (!shadow_abort_topic.empty() && !shadow_rearm_after_internal_rejection)
  {
    publishTransactionStatus(
        "incumbent_rebuild_skipped",
        "reason=process_isolated_shadow;external_incumbent_preserved=1");
    publishTransactionStatus("candidate_rejected", reason);
    const auto phase = map_transaction_state->phase();
    if (map_transaction_state->active() ||
        phase == fast_livo::transactional_mapping::Phase::OVERFLOWED)
      map_transaction_state->markRolledBack();
    if (transaction_uses_delta_voxel_journal &&
        voxelmap_manager->MutationJournalActive(transaction_voxel_journal_id))
      voxelmap_manager->CommitMutationJournal();
    if (transaction_uses_delta_visual_journal &&
        vio_manager->mutationJournalActive(transaction_visual_journal_id))
      vio_manager->commitMutationJournals();
    voxel_map_checkpoint.reset();
    transaction_uses_delta_voxel_journal = false;
    transaction_checkpoint_voxel_count = 0;
    transaction_voxel_journal_id = 0;
    transaction_uses_delta_visual_journal = false;
    transaction_visual_journal_id = 0;
    vio_map_checkpoint.reset();
    map_transaction_buffer.clear();
    replay_incumbent_boundary.reset();
    replay_incumbent_state_valid = false;
    transaction_replay_active = false;
    ros::shutdown();
    return;
  }
  publishTransactionStatus("candidate_rejected", reason);
  if ((!transaction_uses_delta_voxel_journal && !voxel_map_checkpoint) ||
      (transaction_uses_delta_visual_journal &&
       !vio_manager->mutationJournalActive(transaction_visual_journal_id)) ||
      !vio_map_checkpoint || map_transaction_buffer.empty())
  {
    rollbackMapTransaction("candidate rejection lacks verified redo state: " + reason);
    return;
  }
  if (async_replay_payload_spill_enabled)
    replay_payload_journal.commitDurable();

  if (replay_incumbent_boundary && replay_incumbent_boundary->valid &&
      transaction_uses_delta_voxel_journal &&
      transaction_uses_delta_visual_journal)
  {
    const double restore_start = omp_get_wtime();
    const std::size_t buffered_events = map_transaction_buffer.size();
    const double buffered_span_sec = buffered_events > 1 ?
        map_transaction_buffer.back().relative_time -
            map_transaction_buffer.front().relative_time : 0.0;
    publishTransactionStatus(
        "incumbent_exact_restore_started",
        "restore_mode=dirty_after_image;replayed_events=0");

    // First remove every candidate mutation and return to the selected anchor.
    voxelmap_manager->RollbackMutationJournal(transaction_voxel_journal_id);
    vio_manager->rollbackMutationJournal(transaction_visual_journal_id);
    restoreMapTransactionCheckpoint();

    // Then consume the live boundary after-images.  This restores precisely the
    // incumbent generation that existed before replay, without estimator
    // re-execution and its non-bitwise cache/history effects.
    if (!voxelmap_manager->ConsumeMutationRedo(
            *replay_incumbent_boundary->voxel_redo) ||
        !vio_manager->consumeMutationRedo(
            *replay_incumbent_boundary->visual_redo))
    {
      publishTransactionStatus(
          "candidate_terminated",
          "reason=exact_incumbent_after_image_restore_failed");
      ros::shutdown();
      return;
    }

    path.poses.insert(path.poses.end(),
                      replay_incumbent_boundary->path_tail.begin(),
                      replay_incumbent_boundary->path_tail.end());
    for (const auto &entry : replay_incumbent_boundary->trajectory_tail)
      trajectory_poses[entry.first] = entry.second;
    pcl_wait_save->points.insert(
        pcl_wait_save->points.end(),
        replay_incumbent_boundary->pcd_rgb_tail.points.begin(),
        replay_incumbent_boundary->pcd_rgb_tail.points.end());
    pcl_wait_save->width = static_cast<uint32_t>(pcl_wait_save->points.size());
    pcl_wait_save->height = 1;
    pcl_wait_save_intensity->points.insert(
        pcl_wait_save_intensity->points.end(),
        replay_incumbent_boundary->pcd_intensity_tail.points.begin(),
        replay_incumbent_boundary->pcd_intensity_tail.points.end());
    pcl_wait_save_intensity->width =
        static_cast<uint32_t>(pcl_wait_save_intensity->points.size());
    pcl_wait_save_intensity->height = 1;
    restoreMapperCheckpoint(replay_incumbent_boundary->mapper);
    vio_manager->restoreCheckpointMetadata(
        *replay_incumbent_boundary->visual_metadata);
    voxelmap_manager->state_ = _state;

    // Candidate rows were already appended to the audit journal.  Re-append
    // authoritative incumbent rows so canonicalization's last-write-wins rule
    // recovers the public trajectory exactly.
    if (trajectory_journal.is_open())
    {
      for (const BufferedTransactionEvent &event : map_transaction_buffer)
      {
        if (!event.incumbent_state_valid ||
            event.measures.lio_vio_flg != LIO)
          continue;
        const Eigen::Quaterniond q(event.incumbent_state.rot_end);
        trajectory_journal << std::setprecision(15)
            << event.measures.last_lio_update_time << ' '
            << event.incumbent_state.pos_end.x() << ' '
            << event.incumbent_state.pos_end.y() << ' '
            << event.incumbent_state.pos_end.z() << ' '
            << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w()
            << " 4\n";
      }
      trajectory_journal.flush();
    }

    const auto phase = map_transaction_state->phase();
    if (map_transaction_state->active() ||
        phase == fast_livo::transactional_mapping::Phase::OVERFLOWED)
      map_transaction_state->markRolledBack();

    // The consumed rolling journals cannot cover time before this boundary.
    // Seed one exact checkpoint here; subsequent normal events rebuild the
    // configured rollback horizon without retaining the rejected candidate.
    rolling_checkpoints.clear();
    rolling_event_buffer.clear();
    resetReplayPayloadJournal();
    RollingCheckpoint checkpoint;
    checkpoint.relative_time = std::nextafter(
        replay_incumbent_boundary->relative_time,
        -std::numeric_limits<double>::infinity());
    checkpoint.first_event_sequence =
        replay_incumbent_boundary->next_event_sequence;
    checkpoint.voxel_count = voxelmap_manager->voxel_map_.size();
    checkpoint.uses_delta_voxel_journal = true;
    checkpoint.uses_delta_visual_journal = true;
    checkpoint.mapper = captureMapperCheckpoint();
    checkpoint.visual_map = vio_manager->createMetadataCheckpoint();
    checkpoint.voxel_journal_id = voxelmap_manager->BeginMutationJournal();
    checkpoint.visual_journal_id = vio_manager->beginMutationJournal();
    rolling_checkpoints.push_back(std::move(checkpoint));

    voxel_map_checkpoint.reset();
    transaction_uses_delta_voxel_journal = false;
    transaction_checkpoint_voxel_count = 0;
    transaction_voxel_journal_id = 0;
    transaction_uses_delta_visual_journal = false;
    transaction_visual_journal_id = 0;
    vio_map_checkpoint.reset();
    map_transaction_buffer.clear();
    replay_incumbent_state_valid = false;
    transaction_replay_active = false;
    lidar_updates_enabled = true;
    visual_updates_enabled = true;
    lidar_map_writes_enabled = true;
    visual_map_writes_enabled = true;
    voxelmap_manager->selective_residual_gate_active_ = false;
    voxelmap_manager->robust_residual_weight_active_ = false;
    map_transaction_state->resetForMonitoring();
    harm_detector->resetForMonitoring();
    online_detection_pending = false;
    online_point_onset_sec = -1.0;
    online_detected_onset_sec = -1.0;
    online_detection_time_sec = -1.0;
    healthy_validator_limits_ready = false;
    healthy_residual_limit = 0.0;
    healthy_match_ratio_limit = 0.0;
    healthy_match_baseline_reference = 0.0;
    healthy_residual_history.clear();
    healthy_match_history.clear();
    protected_rolling_checkpoint_time_sec = -1.0;
    rolling_checkpoint_rotation_safe = true;
    const bool rejected_attempt_budget_refunded =
        recovery_proposal_arbiter_config.enabled &&
        recovery_proposal_arbiter->refundLastRejectedAttempt();
    int heap_trimmed = 0;
#if defined(__GLIBC__)
    heap_trimmed = malloc_trim(0);
#endif
    const double restore_wall_sec = omp_get_wtime() - restore_start;
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "restore_mode=dirty_after_image"
           << ";incumbent_rebuild_events=0"
           << ";avoided_replay_events=" << buffered_events
           << ";avoided_replay_sensor_span_sec=" << buffered_span_sec
           << ";restore_wall_sec=" << restore_wall_sec
           << ";boundary_capture_wall_sec="
           << replay_incumbent_boundary->capture_wall_sec
           << ";rolling_history_reseeded=1"
           << ";reseed_checkpoint_relative_time_sec="
           << rolling_checkpoints.front().relative_time
           << ";heap_trimmed=" << heap_trimmed
           << ";rejected_attempt_budget_refunded="
           << static_cast<int>(rejected_attempt_budget_refunded);
    publishTransactionStatus("incumbent_exact_restore_completed", detail.str());
    publishTransactionStatus(
        "candidate_rearmed",
        std::string("reason=exact_incumbent_after_image;") + detail.str());
    replay_incumbent_boundary.reset();
    return;
  }

  const std::size_t rebuild_event_count = map_transaction_buffer.size();
  const double rebuild_sensor_span_sec = rebuild_event_count > 1 ?
      map_transaction_buffer.back().relative_time -
          map_transaction_buffer.front().relative_time : 0.0;
  {
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "incumbent_rebuild_events=" << rebuild_event_count
           << ";incumbent_rebuild_sensor_span_sec=" << rebuild_sensor_span_sec;
    publishTransactionStatus("incumbent_rebuild_started", detail.str());
  }
  const double rebuild_total_start = omp_get_wtime();
  const bool reseed_rolling_history =
      shadow_rearm_after_internal_rejection &&
      transaction_uses_delta_voxel_journal &&
      transaction_uses_delta_visual_journal;
  const double rejected_detection_time_sec = online_detection_time_sec;
  std::deque<RollingCheckpoint> rebuilt_rolling_checkpoints;
  std::unique_ptr<fast_livo::harm_detection::OnlineHarmDetector>
      rebuilt_harm_detector;
  if (reseed_rolling_history)
  {
    rebuilt_harm_detector.reset(
        new fast_livo::harm_detection::OnlineHarmDetector(
            harm_detector_config));
  }

  // Restoring the live runtime snapshot is insufficient because VIO contains
  // transient caches whose future behaviour is not bitwise reversible. Rebuild
  // the incumbent from the verified checkpoint and the same raw event journal
  // using the unmodified estimator and full persistent writes.
  const double voxel_restore_start = omp_get_wtime();
  if (transaction_uses_delta_voxel_journal)
    voxelmap_manager->RollbackMutationJournal(transaction_voxel_journal_id);
  else
    voxelmap_manager->RestoreSnapshot(*voxel_map_checkpoint);
  if (transaction_uses_delta_visual_journal)
    vio_manager->rollbackMutationJournal(transaction_visual_journal_id);
  const double voxel_restore_wall_sec = omp_get_wtime() - voxel_restore_start;
  const double state_restore_start = omp_get_wtime();
  restoreMapTransactionCheckpoint();
  const double state_restore_wall_sec = omp_get_wtime() - state_restore_start;
  if (reseed_rolling_history)
  {
    // The rejected candidate consumed the selected rollback journals. Start a
    // fresh anchor at the same verified state, then reconstruct the normal
    // rolling checkpoint grid while rebuilding. A single old anchor leaves a
    // moving platform with almost no overlap against its healthy-tail scans.
    RollingCheckpoint checkpoint;
    checkpoint.relative_time = selected_checkpoint_relative_time_sec;
    checkpoint.first_event_sequence = map_transaction_buffer.front().sequence;
    checkpoint.voxel_count = voxelmap_manager->voxel_map_.size();
    checkpoint.uses_delta_voxel_journal = true;
    checkpoint.uses_delta_visual_journal = true;
    checkpoint.mapper = captureMapperCheckpoint();
    checkpoint.visual_map = vio_manager->createMetadataCheckpoint();
    checkpoint.voxel_journal_id = voxelmap_manager->BeginMutationJournal();
    checkpoint.visual_journal_id = vio_manager->beginMutationJournal();
    rebuilt_rolling_checkpoints.push_back(std::move(checkpoint));
  }
  lidar_updates_enabled = true;
  visual_updates_enabled = true;
  lidar_map_writes_enabled = true;
  visual_map_writes_enabled = true;
  voxelmap_manager->selective_residual_gate_active_ = false;
  voxelmap_manager->robust_residual_weight_active_ = false;
  incumbent_rebuild_trajectory_active = true;
  transaction_replay_active = true;
  const double event_processing_start = omp_get_wtime();
  std::size_t rebuilt_events = 0;
  for (const BufferedTransactionEvent &event : map_transaction_buffer)
  {
    if (reseed_rolling_history && !rebuilt_rolling_checkpoints.empty() &&
        event.relative_time - rebuilt_rolling_checkpoints.back().relative_time >=
            rolling_checkpoint_period_sec)
    {
      RollingCheckpoint checkpoint;
      checkpoint.relative_time = std::nextafter(
          last_completed_event_relative_time,
          -std::numeric_limits<double>::infinity());
      checkpoint.first_event_sequence = event.sequence;
      checkpoint.voxel_count = voxelmap_manager->voxel_map_.size();
      checkpoint.uses_delta_voxel_journal = true;
      checkpoint.uses_delta_visual_journal = true;
      const double checkpoint_total_start = omp_get_wtime();
      const double mapper_copy_start = omp_get_wtime();
      checkpoint.mapper = captureMapperCheckpoint();
      checkpoint.mapper_copy_wall_sec = omp_get_wtime() - mapper_copy_start;
      const double visual_copy_start = omp_get_wtime();
      checkpoint.visual_map = vio_manager->createMetadataCheckpoint();
      checkpoint.visual_copy_wall_sec = omp_get_wtime() - visual_copy_start;
      const double voxel_copy_start = omp_get_wtime();
      checkpoint.voxel_journal_id = voxelmap_manager->BeginMutationJournal();
      checkpoint.voxel_copy_wall_sec = omp_get_wtime() - voxel_copy_start;
      checkpoint.visual_journal_id = vio_manager->beginMutationJournal();
      checkpoint.total_copy_wall_sec =
          omp_get_wtime() - checkpoint_total_start;
      rebuilt_rolling_checkpoints.push_back(std::move(checkpoint));

      const double cutoff = event.relative_time - rolling_history_horizon_sec;
      while (rebuilt_rolling_checkpoints.size() > 1 &&
             rebuilt_rolling_checkpoints[1].relative_time <= cutoff)
      {
        voxelmap_manager->DiscardMutationJournal(
            rebuilt_rolling_checkpoints.front().voxel_journal_id);
        vio_manager->discardMutationJournal(
            rebuilt_rolling_checkpoints.front().visual_journal_id);
        rebuilt_rolling_checkpoints.pop_front();
      }
      const std::size_t retained_checkpoint_limit =
          std::min(rolling_max_checkpoints, rolling_delta_max_checkpoints);
      while (rebuilt_rolling_checkpoints.size() >
             retained_checkpoint_limit)
      {
        voxelmap_manager->DiscardMutationJournal(
            rebuilt_rolling_checkpoints.front().voxel_journal_id);
        vio_manager->discardMutationJournal(
            rebuilt_rolling_checkpoints.front().visual_journal_id);
        rebuilt_rolling_checkpoints.pop_front();
      }
    }
    LidarMeasures = materializeReplayEvent(event);
    processImu();
    stateEstimationAndMapping();
    if (rebuilt_harm_detector)
    {
      const auto rebuilt_decision = rebuilt_harm_detector->observe(
          event.relative_time,
          voxelmap_manager->raw_mean_abs_residual_,
          voxelmap_manager->raw_match_ratio_,
          detector_normalized_innovation);
      if (rebuilt_decision.triggered)
      {
        // Reproduce and consume historical proposals during deterministic
        // incumbent reconstruction. Leaving the detector latched at the
        // rejected proposal would create a blind interval after re-arm.
        rebuilt_harm_detector->resetForMonitoring();
      }
    }
    ++rebuilt_events;
  }
  const double event_processing_wall_sec = omp_get_wtime() - event_processing_start;
  incumbent_rebuild_trajectory_active = false;
  transaction_replay_active = false;
  if (trajectory_journal.is_open())
  {
    for (const auto &entry : trajectory_poses)
    {
      trajectory_journal << std::setprecision(15) << entry.first;
      for (const double value : entry.second) trajectory_journal << ' ' << value;
      trajectory_journal << " 3\n";
    }
    trajectory_journal.flush();
  }

  {
    const double total_wall_sec = omp_get_wtime() - rebuild_total_start;
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "incumbent_rebuild_events=" << rebuilt_events
           << ";incumbent_rebuild_sensor_span_sec=" << rebuild_sensor_span_sec
           << ";incumbent_rebuild_voxel_restore_wall_sec=" << voxel_restore_wall_sec
           << ";incumbent_rebuild_state_visual_restore_wall_sec=" << state_restore_wall_sec
           << ";incumbent_rebuild_event_processing_wall_sec=" << event_processing_wall_sec
           << ";incumbent_rebuild_total_wall_sec=" << total_wall_sec
           << ";incumbent_rebuild_events_per_wall_sec="
           << (event_processing_wall_sec > 0.0 ?
               static_cast<double>(rebuilt_events) / event_processing_wall_sec : 0.0);
    publishTransactionStatus("incumbent_rebuild_completed", detail.str());
  }

  const auto phase = map_transaction_state->phase();
  if (map_transaction_state->active() || phase == fast_livo::transactional_mapping::Phase::OVERFLOWED)
    map_transaction_state->markRolledBack();
  if (rebuilt_harm_detector)
    *harm_detector = std::move(*rebuilt_harm_detector);
  if (reseed_rolling_history)
  {
    rolling_checkpoints.clear();
    rolling_checkpoints.swap(rebuilt_rolling_checkpoints);
    rolling_event_buffer.clear();
    rolling_event_buffer.swap(map_transaction_buffer);
    if (!rolling_checkpoints.empty())
    {
      const std::size_t first_sequence =
          rolling_checkpoints.front().first_event_sequence;
      while (!rolling_event_buffer.empty() &&
             rolling_event_buffer.front().sequence < first_sequence)
        rolling_event_buffer.pop_front();
    }
  }
  voxel_map_checkpoint.reset();
  transaction_uses_delta_voxel_journal = false;
  transaction_checkpoint_voxel_count = 0;
  transaction_voxel_journal_id = 0;
  transaction_uses_delta_visual_journal = false;
  transaction_visual_journal_id = 0;
  vio_map_checkpoint.reset();
  map_transaction_buffer.clear();
  replay_incumbent_boundary.reset();
  replay_incumbent_state_valid = false;
  lidar_updates_enabled = true;
  visual_updates_enabled = true;
  lidar_map_writes_enabled = true;
  visual_map_writes_enabled = true;
  voxelmap_manager->selective_residual_gate_active_ = false;
  voxelmap_manager->robust_residual_weight_active_ = false;
  if (shadow_rearm_after_internal_rejection)
  {
    map_transaction_state->resetForMonitoring();
    harm_detector->resetForMonitoring();
    online_detection_pending = false;
    online_point_onset_sec = -1.0;
    online_detected_onset_sec = -1.0;
    online_detection_time_sec = -1.0;
    healthy_validator_limits_ready = false;
    healthy_residual_limit = 0.0;
    healthy_match_ratio_limit = 0.0;
    healthy_match_baseline_reference = 0.0;
    healthy_residual_history.clear();
    healthy_match_history.clear();
    protected_rolling_checkpoint_time_sec = -1.0;
    rolling_checkpoint_rotation_safe = true;
    if (!reseed_rolling_history)
    {
      rolling_checkpoints.clear();
      rolling_event_buffer.clear();
      resetReplayPayloadJournal();
    }
    int heap_trimmed = 0;
    const bool rejected_attempt_budget_refunded =
        recovery_proposal_arbiter_config.enabled &&
        recovery_proposal_arbiter->refundLastRejectedAttempt();
#if defined(__GLIBC__)
    // Rejection rebuilds release many differently sized octree, image and
    // replay allocations. glibc otherwise keeps those arenas resident, so
    // repeated clean proposals accumulate RSS even though the objects are no
    // longer reachable. Trim only at the coarse re-arm boundary.
    heap_trimmed = malloc_trim(0);
#endif
    publishTransactionStatus(
        "candidate_rearmed",
        std::string("reason=internal_rejection_reconstructed;") +
            "heap_trimmed=" + std::to_string(heap_trimmed) + ";" +
            "rejected_attempt_budget_refunded=" +
            std::to_string(
                static_cast<int>(rejected_attempt_budget_refunded)) + ";" +
            "arbiter_attempts_total=" +
            std::to_string(recovery_proposal_arbiter->attemptsTotal()) + ";" +
            "arbiter_admitted_attempts_lifetime_total=" +
            std::to_string(
                recovery_proposal_arbiter->admittedAttemptsLifetimeTotal()) + ";" +
            "arbiter_rejected_attempt_refunds_total=" +
            std::to_string(
                recovery_proposal_arbiter->rejectedAttemptRefundsTotal()) + ";" +
            "baseline_relearn_required=0;rolling_history_reseeded=" +
            std::to_string(static_cast<int>(reseed_rolling_history)) +
            ";reseed_checkpoint_relative_time_sec=" +
            std::to_string(rolling_checkpoints.empty() ? -1.0 :
                           rolling_checkpoints.front().relative_time) +
            ";reseed_newest_checkpoint_relative_time_sec=" +
            std::to_string(rolling_checkpoints.empty() ? -1.0 :
                           rolling_checkpoints.back().relative_time) +
            ";reseed_checkpoint_count=" +
            std::to_string(rolling_checkpoints.size()) +
            ";reseed_detection_time_sec=" +
            std::to_string(rejected_detection_time_sec));
  }
}

void LIVMapper::preflightPrivateLidarProfile()
{
  // No state/map restore, raw correction, filter or publisher is called here.
  fast_livo::scan_profile::require(!transaction_uses_delta_voxel_journal &&
      !transaction_uses_delta_visual_journal && !map_transaction_buffer.empty(),
      "private_import_required");
  fast_livo::scan_profile::Context actual;
  actual.checkpoint_sha256 = private_candidate_checkpoint_sha256;
  actual.journal_sha256 = fast_livo::source_revision::sha256File(
      private_fork_sealed_journal_path);
  const double checkpoint_absolute_sec = mapper_first_lidar_header_sec +
      map_transaction_checkpoint.last_completed_event_relative_time;
  fast_livo::scan_profile::require(std::isfinite(checkpoint_absolute_sec) &&
      checkpoint_absolute_sec > 0 && checkpoint_absolute_sec < 18446744073.0,
      "checkpoint_time_invalid");
  actual.checkpoint_ns = static_cast<std::uint64_t>(
      std::llround(checkpoint_absolute_sec * 1e9));
  actual.decision_event = map_transaction_buffer.back().sequence;
  actual.decision_ns = p2SensorStampNs(
      materializeReplayEvent(map_transaction_buffer.back()));
  actual.episode = p2_active_episode_id;
  actual.proposal = p2_active_proposal_index;
  private_lidar_profile.beginPreflight(actual);
  for (const auto &event : map_transaction_buffer)
  {
    if (event.measures.lio_vio_flg != LIO) continue;
    const LidarMeasureGroup payload = materializeReplayEvent(event);
    fast_livo::scan_profile::require(payload.raw_reprocess_recipe_valid &&
        !payload.raw_lidar_frames.empty(), "raw_recipe_missing");
    const auto event_ns = p2SensorStampNs(payload);
    for (const auto &frame : payload.raw_lidar_frames)
    {
      std::uint64_t header_ns = 0;
      if (frame.message_type == RawLidarMessageType::POINTCLOUD2)
        header_ns = fast_livo::raw_replay::decode<sensor_msgs::PointCloud2>(frame).header.stamp.toNSec();
      else if (frame.message_type == RawLidarMessageType::LIVOX_CUSTOM)
        header_ns = fast_livo::raw_replay::decode<livox_ros_driver::CustomMsg>(frame).header.stamp.toNSec();
      else throw std::runtime_error("scan_profile:unsupported_raw_type");
      private_lidar_profile.observe(header_ns, frame.serialized_message, event_ns);
    }
  }
  private_lidar_profile.finishPreflight();
}

void LIVMapper::replayMapTransaction()
{
  if ((!transaction_uses_delta_voxel_journal && !voxel_map_checkpoint) ||
      (transaction_uses_delta_voxel_journal &&
       !voxelmap_manager->MutationJournalActive(transaction_voxel_journal_id)) ||
      (transaction_uses_delta_visual_journal &&
       !vio_manager->mutationJournalActive(transaction_visual_journal_id)) ||
      !vio_map_checkpoint || map_transaction_buffer.empty())
  {
    rollbackMapTransaction("missing checkpoint or replay buffer");
    return;
  }

  if (private_fork_require_readonly_journal)
  {
    const BufferedTransactionEvent &event = map_transaction_buffer.front();
    fast_livo::replay::ReplayInputIdentity observed;
    fast_livo::replay::ReplayInputIdentity expected;
    expected.type = private_fork_next_event_type;
    expected.absolute_time_sec = private_fork_next_event_absolute_time_sec;
    expected.cur_hash = private_fork_next_event_cur_hash;
    expected.image_hash = private_fork_next_event_image_hash;
    expected.imu_hash = private_fork_next_event_imu_hash;
    expected.cur_point_count = static_cast<std::size_t>(
        private_fork_next_event_cur_count);
    const double sealed_completed = private_fork_completed_event_time_sec;
    const double checkpoint_completed =
        map_transaction_checkpoint.last_completed_event_relative_time;
    fast_livo::replay::ReplayStartDecision decision;
    decision.accepted = false;
    // One record only. The rest of the journal stays on disk until replay.
    LidarMeasureGroup first_event_payload;
    bool payload_restored = false;
    if (!event.payload_record.spilled || event.payload_record.size == 0 ||
        event.payload_record.checksum == 0)
      decision.reason = "replay_payload_missing";
    else
    {
      try
      {
        first_event_payload = materializeReplayEvent(event);
        payload_restored = true;
      }
      catch (const std::exception &)
      {
        decision.reason = "replay_payload_corrupt";
      }
    }
    if (payload_restored &&
        (!std::isfinite(checkpoint_completed) ||
         std::fabs(checkpoint_completed - sealed_completed) >
             fast_livo::replay::kSensorEpochToleranceSec))
      decision.reason = "checkpoint_completed_time_mismatch";
    else if (payload_restored && first_event_payload.measures.empty())
      decision.reason = "replay_input_missing";
    else if (payload_restored)
    {
      const fast_livo::audit::InputDigest digest =
          fast_livo::audit::inputDigest(first_event_payload);
      const auto &group = first_event_payload.measures.back();
      observed.type = static_cast<int>(first_event_payload.lio_vio_flg);
      observed.absolute_time_sec =
          first_event_payload.lio_vio_flg == VIO ? group.vio_time : group.lio_time;
      observed.cur_hash = digest.cur_hash;
      observed.image_hash = digest.image_hash;
      observed.imu_hash = digest.imu_hash;
      observed.cur_point_count = first_event_payload.pcl_proc_cur ?
          first_event_payload.pcl_proc_cur->size() : 0;
      decision = fast_livo::replay::acceptFirstReplayEvent(
          mapper_first_lidar_header_sec, private_fork_sealed_time_origin_sec,
          sealed_completed, observed, expected);
    }
    const double incumbent_position = _state.pos_end[0];
    const std::size_t incumbent_voxels = voxelmap_manager->voxel_map_.size();
    const bool opened = fast_livo::replay::commitAcceptedReplayStart(
        decision, []() {});
    if (!opened || _state.pos_end[0] != incumbent_position ||
        voxelmap_manager->voxel_map_.size() != incumbent_voxels)
    {
      std::ostringstream detail;
      detail << std::setprecision(17) << decision.reason
             << ";observed_type=" << observed.type
             << ";observed_absolute_time_sec=" << observed.absolute_time_sec
             << ";observed_cur_hash=" << observed.cur_hash
             << ";observed_cur_count=" << observed.cur_point_count
             << ";expected_type=" << expected.type
             << ";expected_absolute_time_sec=" << expected.absolute_time_sec
             << ";expected_cur_hash=" << expected.cur_hash
             << ";expected_cur_count=" << expected.cur_point_count;
      ROS_ERROR_STREAM("[ ReplayStart ] " << detail.str());
      publishTransactionStatus("replay_start_rejected", detail.str(),
                               event.relative_time);
      // The checkpoint object is dropped without being copied onto the live
      // map or estimator. rollbackMapTransaction would apply that copy.
      map_transaction_buffer.clear();
      voxel_map_checkpoint.reset();
      vio_map_checkpoint.reset();
      if (map_transaction_state &&
          (map_transaction_state->active() ||
           map_transaction_state->phase() ==
               fast_livo::transactional_mapping::Phase::OVERFLOWED))
        map_transaction_state->markRolledBack();
      return;
    }
  }

  if (private_lidar_profile_enabled)
  {
    try
    {
      preflightPrivateLidarProfile();
      publishTransactionStatus("private_lidar_profile_admitted",
          "method=" + private_lidar_profile.method() +
          ";public_switch_authorized=0;causal_selection_certified=0",
          map_transaction_buffer.back().relative_time);
    }
    catch (const std::exception &error)
    {
      ROS_ERROR_STREAM("[ PrivateLidarProfile ] " << error.what());
      publishTransactionStatus("replay_start_rejected",
                               "private_lidar_profile_rejected;" + std::string(error.what()),
                               map_transaction_buffer.back().relative_time);
      // Drop imported provisional objects. rollbackMapTransaction would COPY
      // the checkpoint over the live estimator; it must not run on admission failure.
      map_transaction_buffer.clear();
      voxel_map_checkpoint.reset();
      vio_map_checkpoint.reset();
      if (map_transaction_state && (map_transaction_state->active() ||
          map_transaction_state->phase() == fast_livo::transactional_mapping::Phase::OVERFLOWED))
        map_transaction_state->markRolledBack();
      return;
    }
  }

  fast_livo::replay::IngestCarryGuard<LidarMeasureGroup> live_ingest_carry(LidarMeasures);
  const std::size_t replay_event_count = map_transaction_buffer.size();
  const double replay_sensor_span_sec = replay_event_count > 1 ?
      map_transaction_buffer.back().relative_time -
          map_transaction_buffer.front().relative_time : 0.0;
  const double replay_profile_start = omp_get_wtime();
  double replay_voxel_restore_wall_sec = 0.0;
  double replay_state_visual_restore_wall_sec = 0.0;
  double replay_event_processing_wall_sec = 0.0;
  double replay_validation_wall_sec = 0.0;
  std::size_t replay_completed_events = 0;
  std::size_t replay_lio_events = 0;
  std::size_t replay_vio_events = 0;
  std::size_t replay_oracle_corrected_frames = 0;
  std::size_t replay_oracle_corrected_points = 0;
  std::size_t replay_oracle_tagged_points = 0;
  std::size_t replay_oracle_missing_provenance_lio_events = 0;
  double replay_oracle_min_source_time_sec =
      std::numeric_limits<double>::infinity();
  double replay_oracle_max_source_time_sec =
      -std::numeric_limits<double>::infinity();
  double replay_oracle_min_inverse_scale =
      std::numeric_limits<double>::infinity();
  double replay_oracle_max_inverse_scale = 0.0;
  bool replay_early_rejected_by_benefit = false;
  std::string replay_early_rejection_detail;
  try
  {
    // No replay or public decision may consume a merely queued record.
    // This is the checked durable boundary for the entire event batch.
    if (async_replay_payload_spill_enabled)
      replay_payload_journal.commitDurable();
    transaction_replay_active = true;

    replay_incumbent_state = _state;
    replay_incumbent_state_valid = true;
    replay_incumbent_boundary.reset();
    const std::size_t pre_replay_candidate_voxel_count =
        voxelmap_manager->voxel_map_.size();
    const double replay_boundary_time = map_transaction_buffer.back().relative_time;
    logTransactionDiagnostic("replay_pre_restore", replay_boundary_time,
                             &replay_incumbent_state);

    if (shadow_rearm_after_internal_rejection &&
        transaction_uses_delta_voxel_journal &&
        transaction_uses_delta_visual_journal)
    {
      const double boundary_capture_start = omp_get_wtime();
      std::unique_ptr<IncumbentBoundaryImage> boundary(
          new IncumbentBoundaryImage());
      boundary->relative_time = replay_boundary_time;
      boundary->next_event_sequence = next_transaction_event_sequence;
      boundary->mapper = captureMapperCheckpoint();
      boundary->visual_metadata = vio_manager->createMetadataCheckpoint();
      boundary->voxel_redo = voxelmap_manager->CaptureMutationRedo(
          transaction_voxel_journal_id);
      boundary->visual_redo = vio_manager->captureMutationRedo(
          transaction_visual_journal_id);
      if (!boundary->voxel_redo || !boundary->visual_redo)
        throw std::runtime_error(
            "failed to capture exact incumbent boundary after-image");
      if (path.poses.size() < map_transaction_checkpoint.path_pose_count ||
          trajectory_poses.size() <
              map_transaction_checkpoint.trajectory_pose_count ||
          pcl_wait_save->points.size() <
              map_transaction_checkpoint.pcd_rgb_point_count ||
          pcl_wait_save_intensity->points.size() <
              map_transaction_checkpoint.pcd_intensity_point_count)
        throw std::runtime_error(
            "incumbent boundary append-only state predates rollback anchor");
      boundary->path_tail.assign(
          path.poses.begin() + map_transaction_checkpoint.path_pose_count,
          path.poses.end());
      auto trajectory_tail_begin = trajectory_poses.begin();
      std::advance(trajectory_tail_begin,
                   map_transaction_checkpoint.trajectory_pose_count);
      for (auto it = trajectory_tail_begin; it != trajectory_poses.end(); ++it)
        boundary->trajectory_tail.push_back(*it);
      boundary->pcd_rgb_tail.points.assign(
          pcl_wait_save->points.begin() +
              map_transaction_checkpoint.pcd_rgb_point_count,
          pcl_wait_save->points.end());
      boundary->pcd_intensity_tail.points.assign(
          pcl_wait_save_intensity->points.begin() +
              map_transaction_checkpoint.pcd_intensity_point_count,
          pcl_wait_save_intensity->points.end());
      boundary->capture_wall_sec = omp_get_wtime() - boundary_capture_start;
      boundary->valid = true;
      std::ostringstream detail;
      detail << std::setprecision(17)
             << "restore_mode=dirty_after_image"
             << ";capture_wall_sec=" << boundary->capture_wall_sec
             << ";voxel_root_after_images="
             << boundary->voxel_redo->root_after_images.size()
             << ";voxel_node_after_images="
             << boundary->voxel_redo->node_after_images.size()
             << ";visual_bucket_after_images="
             << boundary->visual_redo->bucket_after_images.size()
             << ";path_tail_size=" << boundary->path_tail.size()
             << ";trajectory_tail_size=" << boundary->trajectory_tail.size()
             << ";pcd_rgb_tail_size=" << boundary->pcd_rgb_tail.size()
             << ";pcd_intensity_tail_size="
             << boundary->pcd_intensity_tail.size();
      publishTransactionStatus("incumbent_boundary_captured", detail.str(),
                               replay_boundary_time);
      replay_incumbent_boundary = std::move(boundary);
    }

    const bool validate_candidate = healthy_tail_config.enabled &&
                                    healthy_validator_limits_ready;
    const std::size_t verified_checkpoint_voxel_count =
        transaction_checkpoint_voxel_count;
    const std::size_t delta_before_image_count =
        transaction_uses_delta_voxel_journal ?
        voxelmap_manager->MutationJournalBeforeImageCount(
            transaction_voxel_journal_id) : 0;
    const std::size_t delta_inserted_key_count =
        transaction_uses_delta_voxel_journal ?
        voxelmap_manager->MutationJournalInsertedCount(
            transaction_voxel_journal_id) : 0;
    const std::size_t delta_peak_dirty_count =
        transaction_uses_delta_voxel_journal ?
        voxelmap_manager->MutationJournalPeakDirtyCount(
            transaction_voxel_journal_id) : 0;
    const double delta_clone_wall_sec = transaction_uses_delta_voxel_journal ?
        voxelmap_manager->MutationJournalCloneWallSec(
            transaction_voxel_journal_id) : 0.0;
    const std::size_t visual_delta_before_image_count =
        transaction_uses_delta_visual_journal ?
        vio_manager->mutationJournalBeforeImageCount(
            transaction_visual_journal_id) : 0;
    const std::size_t visual_delta_inserted_key_count =
        transaction_uses_delta_visual_journal ?
        vio_manager->mutationJournalInsertedCount(
            transaction_visual_journal_id) : 0;
    const double visual_delta_clone_wall_sec =
        transaction_uses_delta_visual_journal ?
        vio_manager->mutationJournalCloneWallSec(
            transaction_visual_journal_id) : 0.0;
    {
      std::ostringstream detail;
      detail << std::setprecision(17)
             << "replay_events=" << replay_event_count
             << ";replay_sensor_span_sec=" << replay_sensor_span_sec
             << ";event_clone_count=" << transaction_event_clone_count
             << ";event_clone_total_wall_sec=" << transaction_event_clone_total_wall_sec
             << ";event_clone_maximum_wall_sec=" << transaction_event_clone_maximum_wall_sec
             << ";peak_transaction_buffer_events=" << peak_transaction_buffer_events
             << ";checkpoint_mapper_copy_wall_sec=" << selected_checkpoint_mapper_copy_wall_sec
             << ";checkpoint_voxel_copy_wall_sec=" << selected_checkpoint_voxel_copy_wall_sec
             << ";checkpoint_visual_copy_wall_sec=" << selected_checkpoint_visual_copy_wall_sec
             << ";checkpoint_total_copy_wall_sec=" << selected_checkpoint_total_copy_wall_sec
             << ";delta_before_image_count=" << delta_before_image_count
             << ";delta_inserted_key_count=" << delta_inserted_key_count
             << ";delta_peak_dirty_count=" << delta_peak_dirty_count
             << ";delta_clone_wall_sec=" << delta_clone_wall_sec
             << ";visual_delta_before_image_count="
             << visual_delta_before_image_count
             << ";visual_delta_inserted_key_count="
             << visual_delta_inserted_key_count
             << ";visual_delta_clone_wall_sec="
             << visual_delta_clone_wall_sec
             << ";voxel_restore_mode="
             << (transaction_uses_delta_voxel_journal ? "delta_rollback" :
                 (shadow_abort_topic.empty() ?
                  "reusable_clone" : "ownership_transfer"));
      publishTransactionStatus("replay_started", detail.str(), replay_boundary_time);
    }
    map_transaction_state->markReplayStarted();
    const double voxel_restore_start = omp_get_wtime();
    if (transaction_uses_delta_voxel_journal)
    {
      voxelmap_manager->RollbackMutationJournal(transaction_voxel_journal_id);
      // Candidate replay is another provisional generation.  A same-process
      // mapper needs a second undo generation in case validation rejects it.
      // A process-isolated shadow never reconstructs locally after rejection:
      // the independent incumbent remains authoritative and the candidate
      // exits.  Avoiding redundant replay before-images keeps its peak memory
      // proportional to the single selected-anchor journal.
      if (shadow_abort_topic.empty() || shadow_rearm_after_internal_rejection)
        transaction_voxel_journal_id =
            voxelmap_manager->BeginMutationJournal(
                replay_incumbent_boundary ?
                    replay_incumbent_boundary->voxel_redo.get() : nullptr);
    }
    else if (!shadow_abort_topic.empty())
      voxelmap_manager->ConsumeSnapshot(*voxel_map_checkpoint);
    else
      voxelmap_manager->RestoreSnapshot(*voxel_map_checkpoint);
    if (transaction_uses_delta_visual_journal)
    {
      vio_manager->rollbackMutationJournal(transaction_visual_journal_id);
      if (shadow_abort_topic.empty() || shadow_rearm_after_internal_rejection)
        transaction_visual_journal_id = vio_manager->beginMutationJournal(
            replay_incumbent_boundary ?
                replay_incumbent_boundary->visual_redo.get() : nullptr);
    }
    replay_voxel_restore_wall_sec = omp_get_wtime() - voxel_restore_start;
    const double state_restore_start = omp_get_wtime();
    restoreMapTransactionCheckpoint();
    replay_state_visual_restore_wall_sec = omp_get_wtime() - state_restore_start;
    // Optional P2-only evidence path.  Freeze the verified anchor map before
    // replay so candidate and incumbent poses are scored against identical
    // geometry.  The default remains disabled to preserve frozen P1 behavior.
    std::unique_ptr<VoxelMapSnapshot> common_reference_map;
    if (recovery_benefit_common_reference_map_enabled)
      common_reference_map = voxelmap_manager->CreateSnapshot();
    // Snapshot only the verified anchor's keys.  The pre-replay branch also
    // contains the suspect interval, so its total voxel count is not a valid
    // coverage denominator: successful decontamination is expected to remove
    // some of those keys.  Key retention instead detects destructive loss of
    // already-verified geometry while allowing suspect additions to vanish.
    std::vector<VOXEL_LOCATION> verified_checkpoint_voxel_keys;
    verified_checkpoint_voxel_keys.reserve(voxelmap_manager->voxel_map_.size());
    for (const auto &voxel : voxelmap_manager->voxel_map_)
      verified_checkpoint_voxel_keys.push_back(voxel.first);
    logTransactionDiagnostic("replay_post_restore", map_transaction_buffer.front().relative_time,
                             &map_transaction_checkpoint.state);
    fast_livo::recovery_validation::HealthyTailValidator candidate_validator(healthy_tail_config);
    fast_livo::recovery_validation::Config incumbent_tail_config =
        healthy_tail_config;
    incumbent_tail_config.candidate_summary_required = false;
    fast_livo::recovery_validation::HealthyTailValidator incumbent_validator(
        incumbent_tail_config);
    fast_livo::recovery_validation::RecoveryBenefitValidator benefit_validator(
        recovery_benefit_config);
    if (validate_candidate)
    {
      candidate_validator.arm(map_transaction_state->quarantineEndTime(),
                              healthy_residual_limit, healthy_match_ratio_limit);
      incumbent_validator.arm(map_transaction_state->quarantineEndTime(),
                              healthy_residual_limit, healthy_match_ratio_limit);
    }

    bool logged_quarantine_start = false;
    bool logged_healthy_tail_start = false;
    bool reentry_correction_observed = false;
    double reentry_position_correction_m = 0.0;
    double reentry_rotation_correction_deg = 0.0;
    bool boundary_divergence_observed = false;
    double boundary_position_divergence_m = 0.0;
    double boundary_rotation_divergence_deg = 0.0;
    std::string candidate_validation_detail;
    std::size_t invalid_healthy_tail_observation_count = 0;
    std::size_t common_reference_valid_observation_count = 0;
    std::size_t common_reference_invalid_observation_count = 0;
    std::size_t remaining_paired_benefit_observations = 0;
    for (const BufferedTransactionEvent &event : map_transaction_buffer)
    {
      if (event.measures.lio_vio_flg == LIO &&
          event.relative_time > map_transaction_state->quarantineEndTime() &&
          event.incumbent_health_valid &&
          (!recovery_benefit_common_reference_map_enabled ||
           event.incumbent_state_valid))
        ++remaining_paired_benefit_observations;
    }
    long long replay_index = 0;
    const double event_processing_start = omp_get_wtime();
    for (const BufferedTransactionEvent &event : map_transaction_buffer)
    {
      bool benefit_approval_impossible_after_event = false;
      if (event.measures.lio_vio_flg == LIO)
        ++replay_lio_events;
      else
        ++replay_vio_events;
      LidarMeasures = materializeReplayEvent(event);
      recordP2Fusion("consumed", "production_replay", replay_index + 1,
                     "synced", LidarMeasures);
      fusion_audit.write(replay_index+1, "production_replay", "synced", LidarMeasures, _first_lidar_time, _state, feats_undistort);
      causalDumpCloud("replay", "faulted", event.relative_time, LidarMeasures.pcl_proc_cur);
      if (event.measures.lio_vio_flg == LIO && raw_lidar_reprocess_enabled)
      {
        std::size_t event_corrected_points = 0;
        if (!rebuildProcessedCloudFromRaw(
                LidarMeasures, event.relative_time, event_corrected_points))
          throw std::runtime_error("raw_lidar_reprocess_failed");
        if (event_corrected_points > 0)
        {
          ++replay_oracle_corrected_frames;
          replay_oracle_corrected_points += event_corrected_points;
          for (const RawLidarFrame &frame : LidarMeasures.raw_lidar_frames)
          {
            if (private_lidar_profile_enabled)
            {
              const auto header_ns = frame.message_type == RawLidarMessageType::POINTCLOUD2 ?
                  fast_livo::raw_replay::decode<sensor_msgs::PointCloud2>(frame).header.stamp.toNSec() :
                  fast_livo::raw_replay::decode<livox_ros_driver::CustomMsg>(frame).header.stamp.toNSec();
              const double q = private_lidar_profile.scaleForHeader(header_ns);
              if (q == 1.0) continue;
              const double source_time = static_cast<double>(header_ns) / 1e9 - mapper_first_lidar_header_sec;
              replay_oracle_min_source_time_sec = std::min(replay_oracle_min_source_time_sec, source_time);
              replay_oracle_max_source_time_sec = std::max(replay_oracle_max_source_time_sec, source_time);
              replay_oracle_min_inverse_scale = std::min(replay_oracle_min_inverse_scale, q);
              replay_oracle_max_inverse_scale = std::max(replay_oracle_max_inverse_scale, q);
              continue;  // Do not read injector/oracle tags for this operator.
            }
            const bool estimated_active =
                estimated_range_scale_candidate_enabled &&
                estimated_range_scale_candidate_plan.appliesAt(
                    frame.oracle_source_relative_time);
            const auto oracle_correction = fast_livo::replay_oracle::rawScanCorrection(
                private_raw_scan_oracle_config, frame.oracle_source_relative_time,
                frame.oracle_range_scale_active, frame.oracle_inverse_range_scale);
            if (!estimated_active && !oracle_correction.active) continue;
            const double inverse_scale = estimated_active ?
                estimated_range_scale_candidate_plan.inverseScaleAt(
                    frame.oracle_source_relative_time) :
                oracle_correction.inverse_scale;
            replay_oracle_min_source_time_sec = std::min(
                replay_oracle_min_source_time_sec,
                frame.oracle_source_relative_time);
            replay_oracle_max_source_time_sec = std::max(
                replay_oracle_max_source_time_sec,
                frame.oracle_source_relative_time);
            replay_oracle_min_inverse_scale = std::min(
                replay_oracle_min_inverse_scale,
                inverse_scale);
            replay_oracle_max_inverse_scale = std::max(
                replay_oracle_max_inverse_scale,
                inverse_scale);
          }
          if (replay_oracle_corrected_frames == 1)
          {
            std::ostringstream detail;
            detail << std::setprecision(17)
                   << "correction="
                   << (private_lidar_profile_enabled ?
                       "private_frozen_profile_raw_message_then_reprocess" :
                       estimated_range_scale_candidate_enabled ?
                       "estimated_candidate_raw_message_then_reprocess" :
                       private_raw_scan_oracle_config.enabled ?
                       "private_exact_scan_time_oracle_raw_message_then_reprocess" :
                       "oracle_raw_message_then_reprocess")
                   << ";event_relative_time_sec=" << event.relative_time
                   << ";event_raw_frames="
                   << LidarMeasures.raw_lidar_frames.size()
                   << ";event_corrected_points=" << event_corrected_points
                   << ";processed_points="
                   << LidarMeasures.pcl_proc_cur->size()
                   << ";terminal_range_scale="
                   << (private_lidar_profile_enabled ? 1.0 :
                       estimated_range_scale_candidate_enabled ?
                       estimated_range_scale_candidate_terminal_scale :
                       private_raw_scan_oracle_config.enabled ?
                       private_raw_scan_oracle_config.scale_end :
                       replay_range_scale_oracle_config.scale_end)
                   << ";oracle_interval_consumed="
                   << static_cast<int>(
                       !private_lidar_profile_enabled && !estimated_range_scale_candidate_enabled)
                   << ";terminal_range_scale_applicable=" << static_cast<int>(!private_lidar_profile_enabled);
            publishTransactionStatus("raw_reprocess_correction_started",
                                     detail.str(), event.relative_time);
          }
        }
      }
      if (event.measures.lio_vio_flg == LIO &&
          replay_range_scale_oracle_config.enabled &&
          !raw_lidar_reprocess_enabled)
      {
        std::size_t event_tagged_points = 0;
        std::size_t event_corrected_points = 0;
        auto correct_cloud = [&](PointCloudXYZI::Ptr &cloud)
        {
          if (!cloud) return;
          for (PointType &point : cloud->points)
          {
            if (!fast_livo::replay_oracle::hasRangeScaleProvenance(
                    point.normal_y) ||
                !fast_livo::replay_oracle::validEncodedInverseScale(
                    point.normal_x))
              continue;
            const double inverse_scale = point.normal_x;
            const double source_time_sec = point.normal_z;
            ++event_tagged_points;
            ++replay_oracle_tagged_points;
            replay_oracle_min_source_time_sec = std::min(
                replay_oracle_min_source_time_sec, source_time_sec);
            replay_oracle_max_source_time_sec = std::max(
                replay_oracle_max_source_time_sec, source_time_sec);
            if (replay_range_scale_oracle_config.active(source_time_sec))
            {
              point.x = static_cast<float>(point.x * inverse_scale);
              point.y = static_cast<float>(point.y * inverse_scale);
              point.z = static_cast<float>(point.z * inverse_scale);
              ++event_corrected_points;
              ++replay_oracle_corrected_points;
              replay_oracle_min_inverse_scale = std::min(
                  replay_oracle_min_inverse_scale, inverse_scale);
              replay_oracle_max_inverse_scale = std::max(
                  replay_oracle_max_inverse_scale, inverse_scale);
            }
            point.normal_x = 0.0f;
            point.normal_y = 0.0f;
            point.normal_z = 0.0f;
          }
        };
        correct_cloud(LidarMeasures.pcl_proc_cur);
        if (!imu_en) correct_cloud(LidarMeasures.lidar);
        if (event_tagged_points == 0)
          ++replay_oracle_missing_provenance_lio_events;
        if (event_corrected_points > 0)
        {
          ++replay_oracle_corrected_frames;
          if (replay_oracle_corrected_frames == 1)
          {
            std::ostringstream detail;
            detail << std::setprecision(17)
                   << "oracle=range_scale_inverse_scan_provenance"
                   << ";event_relative_time_sec=" << event.relative_time
                   << ";event_tagged_points=" << event_tagged_points
                   << ";event_corrected_points=" << event_corrected_points
                   << ";source_time_min_sec="
                   << replay_oracle_min_source_time_sec
                   << ";source_time_max_sec="
                   << replay_oracle_max_source_time_sec
                   << ";inverse_scale_min="
                   << replay_oracle_min_inverse_scale
                   << ";inverse_scale_max="
                   << replay_oracle_max_inverse_scale;
            publishTransactionStatus("oracle_replay_correction_started",
                                     detail.str(), event.relative_time);
          }
        }
      }
      causalDumpCloud("replay", "inverse", event.relative_time, LidarMeasures.pcl_proc_cur);
      causalReplaceReplayCloud(event.relative_time);
      fusion_audit.write(replay_index+1, "production_replay", "pre_imu", LidarMeasures, _first_lidar_time, _state, feats_undistort);
      processImu();
      fusion_audit.write(replay_index+1, "production_replay", "post_imu", LidarMeasures, _first_lidar_time, _state, feats_undistort);
      causalDumpCloud("replay", "deskewed", event.relative_time, feats_undistort);

      const bool quarantined = map_transaction_state->isCorruptInterval(event.relative_time);
      lidar_updates_enabled = !quarantined || replay_lidar_updates_in_quarantine;
      visual_updates_enabled = !quarantined || replay_visual_updates_in_quarantine;
      // State estimation and persistent-map mutation are separate privileges.
      // A suspect LiDAR frame may still constrain pose against the verified map,
      // but neither its geometry nor visually coloured derivatives may commit.
      lidar_map_writes_enabled = quarantined ?
          replay_map_writes_in_quarantine :
          replay_map_writes_during_validation_tail;
      visual_map_writes_enabled = quarantined ?
          replay_map_writes_in_quarantine :
          replay_map_writes_during_validation_tail;
      const bool validation_tail = validate_candidate && !quarantined &&
          event.relative_time > map_transaction_state->quarantineEndTime();
      voxelmap_manager->selective_residual_gate_active_ =
          lidar_updates_enabled && selective_lidar_gate_enabled &&
          (quarantined ||
           (validation_tail &&
            selective_lidar_gate_during_validation_tail));
      selective_map_write_gate_active = validation_tail &&
          selective_map_write_gate_during_validation_tail;
      voxelmap_manager->robust_residual_weight_active_ =
          quarantined && lidar_updates_enabled && robust_lidar_weight_enabled;
      const double nominal_depth_error = voxelmap_manager->config_setting_.dept_err_;
      const double nominal_beam_error = voxelmap_manager->config_setting_.beam_err_;
      const bool first_healthy_lio = !reentry_correction_observed &&
          event.measures.lio_vio_flg == LIO &&
          event.relative_time > map_transaction_state->quarantineEndTime();
      const StatesGroup predicted_healthy_state = _state;
      if (quarantined && lidar_updates_enabled && replay_fault_lidar_weight < 1.0)
      {
        const double noise_scale = 1.0 / std::sqrt(replay_fault_lidar_weight);
        voxelmap_manager->config_setting_.dept_err_ *= noise_scale;
        voxelmap_manager->config_setting_.beam_err_ *= noise_scale;
      }
      stateEstimationAndMapping();
      recordP2Fusion("result_ready", "production_replay", replay_index + 1,
                     "post_update", LidarMeasures);
      fusion_audit.write(replay_index+1, "production_replay", "post_update", LidarMeasures, _first_lidar_time, _state, feats_undistort);
      maybeExportTypedEventMapSnapshot("identity_replay", replay_index + 1);
      if (first_healthy_lio)
      {
        reentry_position_correction_m =
            (_state.pos_end - predicted_healthy_state.pos_end).norm();
        const M3D reentry_rotation =
            predicted_healthy_state.rot_end.transpose() * _state.rot_end;
        reentry_rotation_correction_deg =
            Log(reentry_rotation).norm() * 180.0 / M_PI;
        reentry_correction_observed = true;
      }
      voxelmap_manager->selective_residual_gate_active_ = false;
      selective_map_write_gate_active = false;
      voxelmap_manager->robust_residual_weight_active_ = false;
      voxelmap_manager->config_setting_.dept_err_ = nominal_depth_error;
      voxelmap_manager->config_setting_.beam_err_ = nominal_beam_error;
      if (quarantined && !logged_quarantine_start)
      {
        logTransactionDiagnostic("quarantine_start", event.relative_time, nullptr,
                                 replay_index, 1);
        logged_quarantine_start = true;
      }
      if (!quarantined && logged_quarantine_start && !logged_healthy_tail_start &&
          event.relative_time > map_transaction_state->quarantineEndTime())
      {
        logTransactionDiagnostic("healthy_tail_start", event.relative_time, nullptr,
                                 replay_index, 0);
        logged_healthy_tail_start = true;
      }
      logTransactionDiagnostic("replay_frame", event.relative_time, nullptr,
                               replay_index, quarantined ? 1 : 0);
      if (fast_livo::replay::isCommitBoundaryEvent(
              static_cast<std::size_t>(replay_index),
              map_transaction_buffer.size()))
      {
        if (!fast_livo::replay::sameCommitBoundaryTime(
                event.relative_time, replay_boundary_time))
          throw std::runtime_error("replay boundary sensor time mismatch");
        boundary_position_divergence_m =
            (_state.pos_end - replay_incumbent_state.pos_end).norm();
        const M3D boundary_rotation =
            replay_incumbent_state.rot_end.transpose() * _state.rot_end;
        boundary_rotation_divergence_deg =
            Log(boundary_rotation).norm() * 180.0 / M_PI;
        boundary_divergence_observed = true;
        logTransactionDiagnostic("replay_incumbent_aligned", event.relative_time,
                                 replay_incumbent_state_valid ? &replay_incumbent_state : nullptr,
                                 replay_index, quarantined ? 1 : 0);
      }
      ++replay_index;
      if (validate_candidate && event.measures.lio_vio_flg == LIO &&
          event.relative_time > map_transaction_state->quarantineEndTime())
      {
        const bool paired_event_available = event.incumbent_health_valid &&
            (!recovery_benefit_common_reference_map_enabled ||
             event.incumbent_state_valid);
        if (paired_event_available &&
            remaining_paired_benefit_observations > 0)
          --remaining_paired_benefit_observations;
        if (event.incumbent_health_valid)
          incumbent_validator.observe(
              event.relative_time, event.incumbent_raw_residual,
              event.incumbent_raw_match_ratio);
        const double raw_residual = voxelmap_manager->raw_mean_abs_residual_;
        const double raw_match_ratio = voxelmap_manager->raw_match_ratio_;
        if (!std::isfinite(raw_residual) || !std::isfinite(raw_match_ratio) ||
            raw_residual < 0.0 || raw_match_ratio < 0.0 ||
            raw_match_ratio > 1.0)
        {
          ++invalid_healthy_tail_observation_count;
          std::ostringstream invalid;
          invalid << std::setprecision(17)
                  << "invalid_event_index=" << replay_index
                  << ";invalid_event_relative_time_sec=" << event.relative_time
                  << ";invalid_raw_residual=" << raw_residual
                  << ";invalid_raw_match_ratio=" << raw_match_ratio
                  << ";invalid_effective_feature_count="
                  << voxelmap_manager->effct_feat_num_
                  << ";invalid_downsampled_feature_count="
                  << voxelmap_manager->feats_down_size_;
          publishTransactionStatus(
              "invalid_healthy_tail_observation", invalid.str(),
              event.relative_time);
        }
        else
        {
          candidate_validator.observe(event.relative_time,
                                       raw_residual, raw_match_ratio);
          if (paired_event_available)
          {
            if (recovery_benefit_common_reference_map_enabled)
            {
              const auto incumbent_common =
                  voxelmap_manager->EvaluateResidualsAgainstSnapshot(
                      *common_reference_map, event.incumbent_state,
                      voxelmap_manager->feats_down_body_);
              const auto candidate_common =
                  voxelmap_manager->EvaluateResidualsAgainstSnapshot(
                      *common_reference_map, _state,
                      voxelmap_manager->feats_down_body_);
              if (incumbent_common.valid && candidate_common.valid)
              {
                ++common_reference_valid_observation_count;
                benefit_validator.observe(
                    incumbent_common.mean_abs_residual,
                    incumbent_common.match_ratio,
                    candidate_common.mean_abs_residual,
                    candidate_common.match_ratio,
                    event.incumbent_normalized_innovation,
                    detector_normalized_innovation);
              }
              else
              {
                ++common_reference_invalid_observation_count;
                std::ostringstream invalid;
                invalid << std::setprecision(17)
                        << "event_relative_time_sec=" << event.relative_time
                        << ";incumbent_valid=" << incumbent_common.valid
                        << ";incumbent_matched_points="
                        << incumbent_common.matched_points
                        << ";incumbent_total_points="
                        << incumbent_common.total_points
                        << ";candidate_valid=" << candidate_common.valid
                        << ";candidate_matched_points="
                        << candidate_common.matched_points
                        << ";candidate_total_points="
                        << candidate_common.total_points;
                publishTransactionStatus(
                    "invalid_common_reference_observation", invalid.str(),
                    event.relative_time);
              }
            }
            else
            {
              benefit_validator.observe(
                  event.incumbent_raw_residual,
                  event.incumbent_raw_match_ratio,
                  raw_residual, raw_match_ratio,
                  event.incumbent_normalized_innovation,
                  detector_normalized_innovation);
            }
          }
        }
        benefit_approval_impossible_after_event =
            recovery_benefit_config.enabled &&
            !recovery_benefit_allow_absolute_health_dominance &&
            !benefit_validator.approvalStillPossible(
                remaining_paired_benefit_observations);
      }
      ++replay_completed_events;
      if (p2_runtime_receipt.enabled())
        p2_replay_consumed_events = replay_completed_events;
      if (replay_completed_events % 100 == 0 ||
          replay_completed_events == replay_event_count)
      {
        const double elapsed_wall_sec = omp_get_wtime() - event_processing_start;
        const double processed_sensor_span_sec = replay_completed_events > 1 ?
            event.relative_time - map_transaction_buffer.front().relative_time : 0.0;
        std::ostringstream detail;
        detail << std::setprecision(17)
               << "replay_completed_events=" << replay_completed_events
               << ";replay_remaining_events="
               << (replay_event_count - replay_completed_events)
               << ";replay_event_processing_wall_sec=" << elapsed_wall_sec
               << ";replay_processed_sensor_span_sec=" << processed_sensor_span_sec
               << ";replay_events_per_wall_sec="
               << (elapsed_wall_sec > 0.0 ?
                   static_cast<double>(replay_completed_events) / elapsed_wall_sec : 0.0)
               << ";replay_lio_events=" << replay_lio_events
               << ";replay_vio_events=" << replay_vio_events;
        publishTransactionStatus("replay_progress", detail.str(), event.relative_time);
      }
      if (benefit_approval_impossible_after_event)
      {
        replay_early_rejected_by_benefit = true;
        std::ostringstream detail;
        detail << benefit_validator.diagnosticFields()
               << ";paired_benefit_remaining_observations_upper_bound="
               << remaining_paired_benefit_observations
               << ";invalid_healthy_tail_observation_count="
               << invalid_healthy_tail_observation_count
               << ";replay_completed_events=" << replay_completed_events
               << ";replay_total_events=" << replay_event_count;
        replay_early_rejection_detail = detail.str();
        publishTransactionStatus("candidate_benefit_early_rejected",
                                 replay_early_rejection_detail,
                                 event.relative_time);
        throw std::runtime_error(
            "paired_recovery_benefit_approval_impossible;" +
            replay_early_rejection_detail);
      }
    }
    replay_event_processing_wall_sec = omp_get_wtime() - event_processing_start;

    {
      std::ostringstream detail;
      detail << std::setprecision(17)
             << "replay_events=" << replay_completed_events
             << ";replay_lio_events=" << replay_lio_events
             << ";replay_vio_events=" << replay_vio_events
             << ";replay_oracle_corrected_frames="
             << replay_oracle_corrected_frames
             << ";replay_oracle_corrected_points="
             << replay_oracle_corrected_points
             << ";replay_oracle_tagged_points="
             << replay_oracle_tagged_points
             << ";replay_oracle_missing_provenance_lio_events="
             << replay_oracle_missing_provenance_lio_events
             << ";replay_oracle_source_time_min_sec="
             << (std::isfinite(replay_oracle_min_source_time_sec) ?
                 replay_oracle_min_source_time_sec : -1.0)
             << ";replay_oracle_source_time_max_sec="
             << (std::isfinite(replay_oracle_max_source_time_sec) ?
                 replay_oracle_max_source_time_sec : -1.0)
             << ";replay_oracle_inverse_scale_min="
             << (std::isfinite(replay_oracle_min_inverse_scale) ?
                 replay_oracle_min_inverse_scale : 1.0)
             << ";replay_oracle_inverse_scale_max="
             << (replay_oracle_corrected_points > 0 ?
                 replay_oracle_max_inverse_scale : 1.0)
             << ";raw_lidar_reprocess_enabled="
             << static_cast<int>(raw_lidar_reprocess_enabled)
             << ";raw_lidar_reprocess_events="
             << raw_lidar_reprocess_events
             << ";raw_lidar_reprocess_frames="
             << raw_lidar_reprocess_frames
             << ";raw_lidar_reprocess_bytes="
             << raw_lidar_reprocess_bytes
             << ";raw_lidar_reprocess_corrected_points="
             << raw_lidar_reprocess_corrected_points
             << ";raw_lidar_reprocess_output_points="
             << raw_lidar_reprocess_output_points
             << ";raw_lidar_reprocess_failures="
             << raw_lidar_reprocess_failures
             << ";replay_sensor_span_sec=" << replay_sensor_span_sec
             << ";replay_event_processing_wall_sec="
             << replay_event_processing_wall_sec
             << ";candidate_validation_pending="
             << static_cast<int>(validate_candidate);
      // P2 treats replay completion as an execution fact, independent of
      // whether later health/map evidence allows the candidate to commit.
      publishTransactionStatus("replay_completed", detail.str(),
                               replay_boundary_time);
    }

    // Same stage for identity and non-identity branches, before any tail decision.
    exportPrivateCandidateMapReceipt(replay_boundary_time);
    const double validation_start = omp_get_wtime();
    if (validate_candidate)
    {
      publishTransactionStatus(
          "candidate_local_health_decision",
          std::string("locally_healthy=") +
              std::to_string(static_cast<int>(
                  candidate_validator.localHealthApproved())) + ";" +
              candidate_validator.diagnosticFields(),
          replay_boundary_time);
      if (healthy_tail_config.candidate_summary_required &&
          !reentry_correction_observed)
      {
        candidate_validation_detail = candidate_validator.diagnosticFields() +
            ";candidate_summary_missing_reentry=1" +
            ";candidate_summary_missing_boundary=" +
            std::to_string(static_cast<int>(!boundary_divergence_observed));
        publishTransactionStatus(
            "candidate_map_decision",
            "map_acceptable=0;geometric_continuity_acceptable=0;" +
                candidate_validation_detail,
            replay_boundary_time);
        publishTransactionStatus(
            "candidate_commit_decision",
            "commit_eligible=0;commit_dry_run=" +
                std::to_string(static_cast<int>(healthy_tail_commit_dry_run)) +
                ";" + candidate_validation_detail,
            replay_boundary_time);
        exportRejectedCandidateMapEvidence(candidate_validation_detail,
                                            replay_boundary_time);
        throw std::runtime_error("healthy_tail_validation_failed;" +
                                 candidate_validation_detail);
      }
      if (healthy_tail_config.candidate_summary_required &&
          !boundary_divergence_observed)
      {
        candidate_validation_detail = candidate_validator.diagnosticFields() +
            ";candidate_summary_missing_reentry=0" +
            ";candidate_summary_missing_boundary=1";
        publishTransactionStatus(
            "candidate_map_decision",
            "map_acceptable=0;geometric_continuity_acceptable=0;" +
                candidate_validation_detail,
            replay_boundary_time);
        publishTransactionStatus(
            "candidate_commit_decision",
            "commit_eligible=0;commit_dry_run=" +
                std::to_string(static_cast<int>(healthy_tail_commit_dry_run)) +
                ";" + candidate_validation_detail,
            replay_boundary_time);
        exportRejectedCandidateMapEvidence(candidate_validation_detail,
                                            replay_boundary_time);
        throw std::runtime_error("healthy_tail_validation_failed;" +
                                 candidate_validation_detail);
      }
      std::size_t verified_checkpoint_retained_voxel_count = 0;
      for (const VOXEL_LOCATION &key : verified_checkpoint_voxel_keys)
      {
        if (voxelmap_manager->voxel_map_.find(key) !=
            voxelmap_manager->voxel_map_.end())
          ++verified_checkpoint_retained_voxel_count;
      }
      const double map_coverage_ratio =
          verified_checkpoint_voxel_keys.empty() ? 0.0 :
          static_cast<double>(verified_checkpoint_retained_voxel_count) /
              static_cast<double>(verified_checkpoint_voxel_keys.size());
      const double map_support_ratio = healthy_match_baseline_reference > 0.0 ?
          candidate_validator.meanObservedMatchRatio() /
              healthy_match_baseline_reference : 0.0;
      // Candidate and incumbent intentionally represent different estimator
      // histories.  When the output mux is configured to establish an SE(3)
      // gauge at handover, its first aligned candidate pose is exactly
      // coincident with the incumbent boundary and the frozen continuity
      // limits apply to that public pose plus subsequent growth.  Keep the raw
      // branch separation as recovery evidence; do not misclassify it as a
      // public-output discontinuity before the mux alignment is applied.
      const double effective_boundary_position_divergence_m =
          replay_output_boundary_alignment_enabled ? 0.0 :
          boundary_position_divergence_m;
      const double effective_boundary_rotation_divergence_deg =
          replay_output_boundary_alignment_enabled ? 0.0 :
          boundary_rotation_divergence_deg;
      candidate_validator.setCandidateSummary(
          reentry_position_correction_m, reentry_rotation_correction_deg,
          map_coverage_ratio, map_support_ratio,
          effective_boundary_position_divergence_m,
          effective_boundary_rotation_divergence_deg);
      candidate_validation_detail = candidate_validator.diagnosticFields();
      candidate_validation_detail +=
          ";boundary_alignment_delegated_to_output_mux=" +
          std::to_string(static_cast<int>(
              replay_output_boundary_alignment_enabled)) +
          ";raw_boundary_position_divergence_m=" +
          std::to_string(boundary_position_divergence_m) +
          ";raw_boundary_rotation_divergence_deg=" +
          std::to_string(boundary_rotation_divergence_deg) +
          ";effective_boundary_position_divergence_m=" +
          std::to_string(effective_boundary_position_divergence_m) +
          ";effective_boundary_rotation_divergence_deg=" +
          std::to_string(effective_boundary_rotation_divergence_deg);
      candidate_validation_detail += ";" + benefit_validator.diagnosticFields();
      const auto incumbent_tail = incumbent_validator.tailDiagnostics();
      const bool absolute_health_dominance =
          recovery_benefit_allow_absolute_health_dominance &&
          candidate_validator.localHealthApproved() &&
          !incumbent_validator.localHealthApproved();
      const bool recovery_benefit_approved = benefit_validator.approved() ||
          absolute_health_dominance;
      candidate_validation_detail +=
          ";paired_benefit_allow_absolute_health_dominance=" +
          std::to_string(static_cast<int>(
              recovery_benefit_allow_absolute_health_dominance)) +
          ";incumbent_tail_local_health_approved=" +
          std::to_string(static_cast<int>(
              incumbent_validator.localHealthApproved())) +
          ";incumbent_tail_observation_count=" +
          std::to_string(incumbent_tail.observation_count) +
          ";incumbent_tail_residual_pass_rate_lower_bound=" +
          std::to_string(incumbent_tail.residual_pass_rate_lower_bound) +
          ";incumbent_tail_match_pass_rate_lower_bound=" +
          std::to_string(incumbent_tail.match_pass_rate_lower_bound) +
          ";incumbent_tail_joint_pass_rate_lower_bound=" +
          std::to_string(incumbent_tail.joint_pass_rate_lower_bound) +
          ";absolute_health_dominance=" +
          std::to_string(static_cast<int>(absolute_health_dominance)) +
          ";recovery_benefit_approved=" +
          std::to_string(static_cast<int>(recovery_benefit_approved));
      candidate_validation_detail +=
          ";invalid_healthy_tail_observation_count=" +
          std::to_string(invalid_healthy_tail_observation_count);
      candidate_validation_detail +=
          ";paired_benefit_common_reference_map_enabled=" +
          std::to_string(static_cast<int>(
              recovery_benefit_common_reference_map_enabled)) +
          ";paired_benefit_common_reference_voxel_count=" +
          std::to_string(common_reference_map ? common_reference_map->size() : 0) +
          ";paired_benefit_common_reference_valid_observation_count=" +
          std::to_string(common_reference_valid_observation_count) +
          ";paired_benefit_common_reference_invalid_observation_count=" +
          std::to_string(common_reference_invalid_observation_count);
      candidate_validation_detail +=
          ";candidate_boundary_voxel_count=" +
          std::to_string(voxelmap_manager->voxel_map_.size()) +
          ";pre_replay_candidate_voxel_count=" +
          std::to_string(pre_replay_candidate_voxel_count) +
          ";verified_checkpoint_voxel_count=" +
          std::to_string(verified_checkpoint_voxel_count) +
          ";verified_checkpoint_key_count=" +
          std::to_string(verified_checkpoint_voxel_keys.size()) +
          ";verified_checkpoint_retained_voxel_count=" +
          std::to_string(verified_checkpoint_retained_voxel_count);
      publishTransactionStatus(
          "candidate_map_decision",
          std::string("map_acceptable=") +
              std::to_string(static_cast<int>(
                  candidate_validator.mapAcceptable())) +
              ";geometric_continuity_acceptable=" +
              std::to_string(static_cast<int>(
                  candidate_validator.geometricContinuityAcceptable())) +
              ";" + candidate_validation_detail,
          replay_boundary_time);
      const bool commit_eligible = candidate_validator.commitApproved() &&
          recovery_benefit_approved;
      publishTransactionStatus(
          "candidate_commit_decision",
          std::string("commit_eligible=") +
              std::to_string(static_cast<int>(commit_eligible)) +
              ";commit_dry_run=" +
              std::to_string(static_cast<int>(healthy_tail_commit_dry_run)) +
              ";" + candidate_validation_detail,
          replay_boundary_time);
      ROS_WARN_STREAM("[ HealthyTail ] candidate_summary reentry_position_m="
                      << reentry_position_correction_m
                      << " reentry_rotation_deg=" << reentry_rotation_correction_deg
                      << " map_coverage_ratio=" << map_coverage_ratio
                      << " map_support_ratio=" << map_support_ratio
                      << " boundary_position_divergence_m="
                      << effective_boundary_position_divergence_m
                      << " boundary_rotation_divergence_deg="
                      << effective_boundary_rotation_divergence_deg
                      << " raw_boundary_position_divergence_m="
                      << boundary_position_divergence_m
                      << " raw_boundary_rotation_divergence_deg="
                      << boundary_rotation_divergence_deg
                      << " tail_approved=" << candidate_validator.approved()
                      << " commit_approved=" << commit_eligible
                      << " benefit_approved=" << recovery_benefit_approved
                      << " absolute_health_dominance="
                      << absolute_health_dominance
                      << " failed_checks=" << candidate_validator.failedChecks());
      if (!commit_eligible)
      {
        exportRejectedCandidateMapEvidence(candidate_validation_detail,
                                            replay_boundary_time);
        throw std::runtime_error("healthy_tail_validation_failed;" +
                                 candidate_validation_detail);
      }
    }
    replay_validation_wall_sec = omp_get_wtime() - validation_start;

    {
      const double total_wall_sec = omp_get_wtime() - replay_profile_start;
      std::ostringstream detail;
      detail << std::setprecision(17)
             << "replay_events=" << replay_completed_events
             << ";replay_lio_events=" << replay_lio_events
             << ";replay_vio_events=" << replay_vio_events
             << ";replay_oracle_corrected_frames="
             << replay_oracle_corrected_frames
             << ";replay_oracle_corrected_points="
             << replay_oracle_corrected_points
             << ";replay_oracle_tagged_points="
             << replay_oracle_tagged_points
             << ";replay_oracle_missing_provenance_lio_events="
             << replay_oracle_missing_provenance_lio_events
             << ";replay_oracle_source_time_min_sec="
             << (std::isfinite(replay_oracle_min_source_time_sec) ?
                 replay_oracle_min_source_time_sec : -1.0)
             << ";replay_oracle_source_time_max_sec="
             << (std::isfinite(replay_oracle_max_source_time_sec) ?
                 replay_oracle_max_source_time_sec : -1.0)
             << ";replay_oracle_inverse_scale_min="
             << (std::isfinite(replay_oracle_min_inverse_scale) ?
                 replay_oracle_min_inverse_scale : 1.0)
             << ";replay_oracle_inverse_scale_max="
             << (replay_oracle_corrected_points > 0 ?
                 replay_oracle_max_inverse_scale : 1.0)
             << ";raw_lidar_reprocess_enabled="
             << static_cast<int>(raw_lidar_reprocess_enabled)
             << ";raw_lidar_reprocess_events="
             << raw_lidar_reprocess_events
             << ";raw_lidar_reprocess_frames="
             << raw_lidar_reprocess_frames
             << ";raw_lidar_reprocess_bytes="
             << raw_lidar_reprocess_bytes
             << ";raw_lidar_reprocess_corrected_points="
             << raw_lidar_reprocess_corrected_points
             << ";raw_lidar_reprocess_output_points="
             << raw_lidar_reprocess_output_points
             << ";raw_lidar_reprocess_failures="
             << raw_lidar_reprocess_failures
             << ";replay_sensor_span_sec=" << replay_sensor_span_sec
             << ";replay_voxel_restore_wall_sec=" << replay_voxel_restore_wall_sec
             << ";replay_state_visual_restore_wall_sec="
             << replay_state_visual_restore_wall_sec
             << ";replay_event_processing_wall_sec=" << replay_event_processing_wall_sec
             << ";replay_validation_wall_sec=" << replay_validation_wall_sec
             << ";replay_total_wall_sec=" << total_wall_sec
             << ";replay_events_per_wall_sec="
             << (replay_event_processing_wall_sec > 0.0 ?
                 static_cast<double>(replay_completed_events) /
                     replay_event_processing_wall_sec : 0.0)
             << ";replay_payload_journal_bytes="
             << replay_payload_journal.bytesWritten()
             << ";replay_payload_durable_bytes="
             << replay_payload_journal.durableBytes()
             << ";replay_payload_peak_queued_bytes="
             << replay_payload_journal.peakQueuedBytes()
             << ";replay_payload_backpressure_wall_sec="
             << replay_payload_journal.backpressureWallSec()
             << ";replay_payload_records_read="
             << replay_payload_journal.recordsRead()
             << ";replay_payload_read_wall_sec="
             << replay_payload_journal.readWallSec();
      publishTransactionStatus("replay_profile_finalized", detail.str(),
                               replay_boundary_time);
    }

    logTransactionDiagnostic("replay_boundary_forward_step", replay_boundary_time,
                             replay_incumbent_state_valid ? &replay_incumbent_state : nullptr,
                             static_cast<long long>(map_transaction_buffer.size()) - 1, 0);

    if (replay_reanchor_pose_to_incumbent_at_commit &&
        replay_incumbent_state_valid)
    {
      const M3D correction_rotation =
          replay_incumbent_state.rot_end * _state.rot_end.transpose();
      const V3D correction_translation = replay_incumbent_state.pos_end -
          correction_rotation * _state.pos_end;
      const double correction_position_m =
          (_state.pos_end - replay_incumbent_state.pos_end).norm();
      const double correction_rotation_deg =
          Log(correction_rotation).norm() * 180.0 / M_PI;
      _state.rot_end = correction_rotation * _state.rot_end;
      _state.pos_end = correction_rotation * _state.pos_end +
          correction_translation;
      _state.vel_end = correction_rotation * _state.vel_end;
      _state.gravity = correction_rotation * _state.gravity;
      MD(DIM_STATE, DIM_STATE) frame_jacobian =
          MD(DIM_STATE, DIM_STATE)::Identity();
      frame_jacobian.block<3, 3>(3, 3) = correction_rotation;
      frame_jacobian.block<3, 3>(7, 7) = correction_rotation;
      frame_jacobian.block<3, 3>(16, 16) = correction_rotation;
      _state.cov = frame_jacobian * _state.cov * frame_jacobian.transpose();
      state_propagat = _state;
      latest_ekf_state = _state;
      imu_propagate = _state;
      voxelmap_manager->state_ = _state;
      std::ostringstream detail;
      detail << std::setprecision(17)
             << "position_correction_m=" << correction_position_m
             << ";rotation_correction_deg=" << correction_rotation_deg
             << ";validation_tail_map_writes_enabled="
             << static_cast<int>(replay_map_writes_during_validation_tail);
      publishTransactionStatus("candidate_pose_reanchored", detail.str(),
                               replay_boundary_time);
    }

    transaction_replay_active = false;
    lidar_updates_enabled = true;
    visual_updates_enabled = true;
    lidar_map_writes_enabled = true;
    visual_map_writes_enabled = true;
    if (transaction_uses_delta_voxel_journal &&
        voxelmap_manager->MutationJournalActive())
      voxelmap_manager->CommitMutationJournal();
    if (transaction_uses_delta_visual_journal &&
        vio_manager->mutationJournalActive())
      vio_manager->commitMutationJournals();
    map_transaction_state->markCommitted();
    logTransactionDiagnostic(healthy_tail_commit_dry_run ?
                             "commit_dry_run_internal" : "commit",
                             replay_boundary_time, nullptr);
    if (healthy_tail_commit_dry_run)
      publishTransactionStatus(
          "candidate_commit_dry_run_eligible",
          "commit_eligible=1;public_generation_switched=0;private_candidate_continues=1;" +
              candidate_validation_detail,
          replay_boundary_time);
    else
      publishTransactionStatus("candidate_committed", candidate_validation_detail,
                               replay_boundary_time);
    ROS_WARN_STREAM("[ Transaction ] Replay committed " << map_transaction_buffer.size()
                    << " buffered events; restored map now contains "
                    << voxelmap_manager->voxel_map_.size() << " voxels"
                    << (validate_candidate ? "; candidate healthy-tail validated" : ""));
    voxel_map_checkpoint.reset();
    transaction_uses_delta_voxel_journal = false;
    transaction_checkpoint_voxel_count = 0;
    transaction_voxel_journal_id = 0;
    transaction_uses_delta_visual_journal = false;
    transaction_visual_journal_id = 0;
    vio_map_checkpoint.reset();
    map_transaction_buffer.clear();
    replay_incumbent_boundary.reset();
    replay_incumbent_state_valid = false;
  }
  catch (const std::exception &error)
  {
    if (p2_current_fusion_pending &&
        p2_current_route == "production_replay")
      recordP2Fusion("disposition", "production_replay",
                     p2_current_event_id, "replay_exception", LidarMeasures);
    if (replay_event_processing_wall_sec <= 0.0 && replay_completed_events > 0)
      replay_event_processing_wall_sec = omp_get_wtime() - replay_profile_start -
          replay_voxel_restore_wall_sec - replay_state_visual_restore_wall_sec;
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "replay_completed_events=" << replay_completed_events
           << ";replay_total_events=" << replay_event_count
           << ";replay_voxel_restore_wall_sec=" << replay_voxel_restore_wall_sec
           << ";replay_state_visual_restore_wall_sec="
           << replay_state_visual_restore_wall_sec
           << ";replay_event_processing_wall_sec=" << replay_event_processing_wall_sec
           << ";replay_total_wall_sec=" << (omp_get_wtime() - replay_profile_start);
    const bool replay_execution_completed =
        replay_completed_events == replay_event_count && replay_event_count > 0;
    publishTransactionStatus(
        replay_early_rejected_by_benefit ?
            "candidate_validation_early_rejected_profile" :
        replay_execution_completed ? "candidate_validation_rejected_profile" :
                                     "replay_failed_profile",
        detail.str());
    rejectCandidateAndRestoreIncumbent(
        std::string(replay_early_rejected_by_benefit ?
                    "candidate benefit early rejected: " :
                    replay_execution_completed ?
                    "candidate validation rejected: " : "replay failed: ") +
        error.what());
  }
}

void LIVMapper::run() 
{
  ros::Rate rate(5000);
  while (ros::ok()) 
  {
    ros::spinOnce();
    if (source_revision_cut_enabled)
      maybeWriteSourceRevisionCut();
    if (!sync_packages(LidarMeasures)) 
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    const MeasureGroup &latest_measure = LidarMeasures.measures.back();
    const double event_time = LidarMeasures.lio_vio_flg == VIO ? latest_measure.vio_time : latest_measure.lio_time;
    const double relative_time = event_time - _first_lidar_time;
    if (private_frozen_scale_probe_enabled &&
        !private_frozen_scale_probe_map_frozen && lidar_map_inited &&
        relative_time >= private_frozen_scale_probe_reference_sec)
    {
      std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] before freeze t=%.6f\n",
                   relative_time);
      std::fflush(stderr);
      if (private_frozen_scale_probe_snapshot_reference_enabled)
      {
        // A private immutable reference does not suppress normal map updates
        // or mutate the incumbent/public estimator after the reference time.
        private_frozen_scale_probe_reference_snapshot =
            voxelmap_manager->CreateSnapshot();
        if (!private_frozen_scale_probe_reference_snapshot)
          throw std::runtime_error("private frozen-map snapshot failed");
        ROS_INFO_STREAM("[ PrivateFrozenScaleProbe ] private reference snapshot captured before event t="
                        << relative_time << " requested="
                        << private_frozen_scale_probe_reference_sec);
      }
      else
      {
        ROS_INFO_STREAM("[ PrivateFrozenScaleProbe ] diagnostic live map writes frozen before event t="
                        << relative_time << " requested="
                        << private_frozen_scale_probe_reference_sec);
      }
      private_frozen_scale_probe_map_frozen = true;
    }
    ++fusion_audit_event;
    fast_livo::audit::InputEventProfile::Scope fusion_profile_scope(
        input_event_profile, "fusion", fusion_audit_event, 0,
        static_cast<std::uint64_t>(event_time * 1e9), 0, relative_time);
    recordP2Fusion("consumed", "direct", fusion_audit_event,
                   "synced", LidarMeasures);
    fusion_audit.write(fusion_audit_event, "direct", "synced", LidarMeasures, _first_lidar_time, _state, feats_undistort);
    nominalProbeBeforeEvent(relative_time);
    causalArchiveDirectEvent(relative_time);
    causalDumpCloud("direct", "input", relative_time, LidarMeasures.pcl_proc_cur);
    processPendingExternalRecoveryProposal(relative_time);
    // The public incumbent exports immutable seeds only at checkpoint cadence.
    // It does not start replay or a candidate process on clean sequences.
    maybeExportRecoveryHandoff(relative_time);
    fast_livo::transactional_mapping::Decision transaction_decision;
    bool event_promoted_from_rolling_history = false;
    if (rolling_history_enabled && map_transaction_config.enabled &&
        map_transaction_state->phase() == fast_livo::transactional_mapping::Phase::MONITORING)
    {
      maintainRollingHistory(relative_time);
      bool detection_due = false;
      if (online_harm_detector_enabled)
      {
        if (online_detection_pending)
        {
          map_transaction_state->setRuntimeInterval(
              online_detected_onset_sec,
              online_detection_time_sec + online_quarantine_duration_sec,
              online_detection_time_sec);
          detection_due = true;
        }
      }
      else
      {
        detection_due = relative_time >= map_transaction_state->detectionTime();
      }
      if (detection_due && !diagnostic_forbid_recovery)
      {
        transaction_decision = map_transaction_state->advance(relative_time);
        if (transaction_decision.begin)
        {
          event_promoted_from_rolling_history = activateRollingTransaction(relative_time);
          online_detection_pending = false;
        }
      }
    }
    else if (!diagnostic_forbid_recovery)
    {
      transaction_decision = map_transaction_state->advance(relative_time);
      if (transaction_decision.begin) beginMapTransaction(relative_time);
    }
    if (transaction_decision.detection_confirmed)
    {
      ROS_WARN_STREAM("[ Transaction ] Detection confirmed at t=" << relative_time
                      << " s (estimated onset=" << map_transaction_state->quarantineStartTime()
                      << " s, quarantine end=" << map_transaction_state->quarantineEndTime()
                      << " s, replay boundary=" << map_transaction_state->replayTime() << " s)");
    }
    if (transaction_decision.buffer && map_transaction_state->active() &&
        !event_promoted_from_rolling_history)
    {
      if (!bufferMapTransactionEvent(relative_time)) {
        recordP2Fusion("disposition", "direct", fusion_audit_event,
                       "buffer_rejected", LidarMeasures);
        fusion_profile_scope.disposition("buffer_rejected");
        continue;
      }
    }
    if (transaction_decision.trigger_replay &&
        map_transaction_state->phase() == fast_livo::transactional_mapping::Phase::REPLAY_READY)
    {
      recordP2Fusion("disposition", "direct", fusion_audit_event,
                     "transferred_to_replay", LidarMeasures);
      replayMapTransaction();
      fusion_profile_scope.disposition("replay_triggered");
      continue;
    }

    if (incremental_public_guard.consumeExpiry(relative_time))
      publishTransactionStatus("incremental_public_guard_expired",
          "reason=bounded_sensor_time_horizon", relative_time);
    const bool incremental_guard_active =
        incremental_public_guard.active(relative_time);
    const bool quarantining = map_transaction_state->phase() == fast_livo::transactional_mapping::Phase::QUARANTINING;
    // A conservative mode freezes VIO on the provisional branch. With a deep
    // visual-map checkpoint, experiments can instead advance and contaminate
    // the complete live branch, then restore its landmark graph before replay.
    visual_updates_enabled = !quarantining || !freeze_vio_provisional;
    lidar_updates_enabled = true;
    lidar_map_writes_enabled =
        (!quarantining || live_lidar_provisional_writes) &&
        !incremental_guard_active &&
        !(private_frozen_scale_probe_map_frozen &&
          !private_frozen_scale_probe_snapshot_reference_enabled);
    visual_map_writes_enabled =
        (!quarantining || !freeze_vio_provisional) &&
        !incremental_guard_active &&
        !(private_frozen_scale_probe_map_frozen &&
          !private_frozen_scale_probe_snapshot_reference_enabled);

    fusion_audit.write(fusion_audit_event, "direct", "pre_imu", LidarMeasures, _first_lidar_time, _state, feats_undistort);
    const bool private_scale_probe_due =
        private_frozen_scale_probe_enabled &&
        private_frozen_scale_probe_map_frozen &&
        LidarMeasures.lio_vio_flg == LIO &&
        relative_time <= private_frozen_scale_probe_end_sec &&
        (private_frozen_scale_probe_last_sec < 0.0 ||
         relative_time - private_frozen_scale_probe_last_sec >=
             private_frozen_scale_probe_period_sec);
    if (private_scale_probe_due)
    {
      std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] before pre-imu checkpoint t=%.6f\n",
                   relative_time);
      std::fflush(stderr);
    }
    const StatesGroup private_probe_state_before_imu =
        private_scale_probe_due ? _state : StatesGroup();
    const ImuProcess::Checkpoint private_probe_imu_before_event =
        private_scale_probe_due ? p_imu->createCheckpoint() :
                                  ImuProcess::Checkpoint();
    processImu();
    if (private_scale_probe_due)
    {
      std::fprintf(stderr, "[ PrivateFrozenScaleProbe ] after primary imu t=%.6f\n",
                   relative_time);
      std::fflush(stderr);
    }
    if (private_scale_probe_due)
    {
      runPrivateFrozenScaleProbe(relative_time,
                                 private_probe_state_before_imu,
                                 private_probe_imu_before_event);
      private_frozen_scale_probe_last_sec = relative_time;
    }
    fusion_audit.write(fusion_audit_event, "direct", "post_imu", LidarMeasures, _first_lidar_time, _state, feats_undistort);
    causalDumpCloud("direct", "deskewed", relative_time, feats_undistort);

    // if (!p_imu->imu_time_init) continue;

    const double guard_nominal_depth_error =
        voxelmap_manager->config_setting_.dept_err_;
    const double guard_nominal_beam_error =
        voxelmap_manager->config_setting_.beam_err_;
    if (incremental_guard_active &&
        LidarMeasures.lio_vio_flg == LIO)
    {
      const double noise_scale = 1.0 / std::sqrt(
          incremental_public_guard.lidarInformationWeight());
      voxelmap_manager->config_setting_.dept_err_ *= noise_scale;
      voxelmap_manager->config_setting_.beam_err_ *= noise_scale;
    }
    stateEstimationAndMapping();
    voxelmap_manager->config_setting_.dept_err_ = guard_nominal_depth_error;
    voxelmap_manager->config_setting_.beam_err_ = guard_nominal_beam_error;

    recordP2Fusion("result_ready", "direct", fusion_audit_event,
                   "post_update", LidarMeasures);
    fusion_audit.write(fusion_audit_event, "direct", "post_update", LidarMeasures, _first_lidar_time, _state, feats_undistort);
    maybeExportTypedEventMapSnapshot("direct", fusion_audit_event);
    if (nominalProbeAfterEvent(relative_time)) {
      fusion_profile_scope.disposition("nominal_probe_stop");
      break;
    }

    if (LidarMeasures.lio_vio_flg == LIO)
    {
      BufferedTransactionEvent *incumbent_event = nullptr;
      if (map_transaction_state->active() && !map_transaction_buffer.empty() &&
          std::abs(map_transaction_buffer.back().relative_time - relative_time) <= 1e-6)
        incumbent_event = &map_transaction_buffer.back();
      else if (!rolling_event_buffer.empty() &&
               std::abs(rolling_event_buffer.back().relative_time - relative_time) <= 1e-6)
        incumbent_event = &rolling_event_buffer.back();
      if (incumbent_event)
      {
        incumbent_event->incumbent_raw_residual =
            voxelmap_manager->raw_mean_abs_residual_;
        incumbent_event->incumbent_raw_match_ratio =
            voxelmap_manager->raw_match_ratio_;
        incumbent_event->incumbent_normalized_innovation =
            detector_normalized_innovation;
        incumbent_event->incumbent_state = _state;
        incumbent_event->incumbent_state_valid =
            _state.rot_end.allFinite() && _state.pos_end.allFinite() &&
            _state.cov.allFinite();
        incumbent_event->incumbent_health_valid =
            std::isfinite(incumbent_event->incumbent_raw_residual) &&
            std::isfinite(incumbent_event->incumbent_raw_match_ratio) &&
            incumbent_event->incumbent_raw_residual >= 0.0 &&
            incumbent_event->incumbent_raw_match_ratio >= 0.0 &&
            incumbent_event->incumbent_raw_match_ratio <= 1.0;
      }
    }

    lidar_map_writes_enabled = true;
    visual_map_writes_enabled = true;
    lidar_updates_enabled = true;
    visual_updates_enabled = true;
    fusion_profile_scope.disposition("completed");
  }
  if (async_replay_payload_spill_enabled)
    replay_payload_journal.commitDurable();
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(extR * p_body_lidar + extT);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  fast_livo::audit::InputEventProfile::Scope input_profile_scope(
      input_event_profile, "lidar", ++input_profile_lidar_sequence,
      msg->header.seq, msg->header.stamp.toNSec(), msg->data.size());
  if (!lidar_en) {
    input_profile_scope.disposition("disabled");
    return;
  }
  mtx_buffer.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  if (!std::isfinite(mapper_first_lidar_header_sec))
    mapper_first_lidar_header_sec = cur_head_time;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
    lid_header_time_buffer.clear();
    lid_raw_message_buffer.clear();
    raw_lidar_carry_frames.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  annotateOracleRangeScaleProvenance(ptr, msg->header.stamp.toSec());
  if (raw_lidar_reprocess_enabled)
  {
    if (!std::isfinite(raw_lidar_reprocess_epoch_sensor_time_sec))
      raw_lidar_reprocess_epoch_sensor_time_sec = msg->header.stamp.toSec();
    const double relative_sec = msg->header.stamp.toSec() -
        raw_lidar_reprocess_epoch_sensor_time_sec;
    lid_raw_message_buffer.push_back(fast_livo::raw_replay::capture(
        *msg, RawLidarMessageType::POINTCLOUD2, cur_head_time, ptr,
        relative_sec,
        replay_range_scale_oracle_config.inverseScale(relative_sec),
        replay_range_scale_oracle_config.active(relative_sec)));
  }
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
  input_profile_scope.disposition("buffered");
}

void LIVMapper::annotateOracleRangeScaleProvenance(
    PointCloudXYZI::Ptr &cloud, const double sensor_header_time_sec)
{
  if (!replay_range_scale_oracle_config.enabled || !cloud || cloud->empty())
    return;
  if (!std::isfinite(replay_range_scale_oracle_epoch_sensor_time_sec))
  {
    replay_range_scale_oracle_epoch_sensor_time_sec = sensor_header_time_sec;
    ROS_WARN_STREAM("[ ReplayOracle ] source-scan epoch=" << std::fixed
                    << std::setprecision(9)
                    << replay_range_scale_oracle_epoch_sensor_time_sec);
  }

  const double relative_sec = sensor_header_time_sec -
      replay_range_scale_oracle_epoch_sensor_time_sec;
  const float inverse_scale = static_cast<float>(
      replay_range_scale_oracle_config.inverseScale(relative_sec));
  for (PointType &point : cloud->points)
  {
    point.normal_x = inverse_scale;
    point.normal_y =
        fast_livo::replay_oracle::kRangeScaleProvenanceMarker;
    point.normal_z = static_cast<float>(relative_sec);
  }
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  fast_livo::audit::InputEventProfile::Scope input_profile_scope(
      input_event_profile, "lidar", ++input_profile_lidar_sequence,
      msg_in->header.seq, msg_in->header.stamp.toNSec(),
      msg_in->points.size() * sizeof(msg_in->points[0]));
  if (!lidar_en) {
    input_profile_scope.disposition("disabled");
    return;
  }
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  if (!std::isfinite(mapper_first_lidar_header_sec))
    mapper_first_lidar_header_sec = cur_head_time;
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
    lid_header_time_buffer.clear();
    lid_raw_message_buffer.clear();
    raw_lidar_carry_frames.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  annotateOracleRangeScaleProvenance(ptr, msg->header.stamp.toSec());

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    input_profile_scope.disposition("empty_cloud");
    return;
  }

  if (raw_lidar_reprocess_enabled)
  {
    if (!std::isfinite(raw_lidar_reprocess_epoch_sensor_time_sec))
      raw_lidar_reprocess_epoch_sensor_time_sec = msg->header.stamp.toSec();
    const double relative_sec = msg->header.stamp.toSec() -
        raw_lidar_reprocess_epoch_sensor_time_sec;
    lid_raw_message_buffer.push_back(fast_livo::raw_replay::capture(
        *msg, RawLidarMessageType::LIVOX_CUSTOM, cur_head_time, ptr,
        relative_sec,
        replay_range_scale_oracle_config.inverseScale(relative_sec),
        replay_range_scale_oracle_config.active(relative_sec)));
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
  input_profile_scope.disposition("buffered");
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  fast_livo::audit::InputEventProfile::Scope input_profile_scope(
      input_event_profile, "imu", ++input_profile_imu_sequence,
      msg_in->header.seq, msg_in->header.stamp.toNSec(), 0);
  if (!imu_en) {
    input_profile_scope.disposition("disabled");
    return;
  }

  if (last_timestamp_lidar < 0.0) {
    input_profile_scope.disposition("before_lidar");
    return;
  }
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    input_profile_scope.disposition("loopback");
    return;
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
  //   mtx_buffer.unlock();
  //   sig_buffer.notify_all();
  //   return;
  // }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
  input_profile_scope.disposition("buffered");
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  fast_livo::audit::InputEventProfile::Scope input_profile_scope(
      input_event_profile, "image", ++input_profile_image_sequence,
      msg_in->header.seq, msg_in->header.stamp.toNSec(), msg_in->data.size());
  if (!img_en) {
    input_profile_scope.disposition("disabled");
    return;
  }
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_img);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  if (hilti_en)
  {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0) {
      input_profile_scope.disposition("hilti_decimated");
      return;
    }
  }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001) {
    input_profile_scope.disposition("duplicate_stamp");
    return;
  }
  ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) {
    input_profile_scope.disposition("before_lidar");
    return;
  }

  if (msg_header_time < last_timestamp_img)
  {
    ROS_ERROR("image loop back. \n");
    input_profile_scope.disposition("loopback");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02)
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    input_profile_scope.disposition("short_interval");
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
  input_profile_scope.disposition("buffered");
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  // The LIO half already consumed the IMU/LiDAR needed for its paired image.
  // Requiring a future scan/IMU here strands the final VIO event at bag end
  // and makes completion depend on callback/processing timing.
  const bool paired_vio_pending = slam_mode_ == LIVO && meas.lio_vio_flg == LIO;
  if (lid_raw_data_buffer.empty() && lidar_en && !paired_vio_pending) return false;
  if (img_buffer.empty() && img_en) return false;
  if (imu_buffer.empty() && imu_en && !paired_vio_pending) return false;

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      initializeRawReplayRecipe(meas, meas.lidar_frame_beg_time,
                                meas.lidar_frame_end_time, false);
      if (raw_lidar_reprocess_enabled)
      {
        if (lid_raw_message_buffer.empty())
          throw std::runtime_error("raw LiDAR buffer missing ONLY_LIO scan");
        meas.raw_lidar_frames.push_back(lid_raw_message_buffer.front());
      }
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    if (raw_lidar_reprocess_enabled) lid_raw_message_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = imu_buffer.back()->header.stamp.toSec();

      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // ROS_ERROR("lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (imu_buffer.front()->header.stamp.toSec() > m.lio_time) break;

        if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // imu_buffer.front()->header.stamp.toSec());
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      initializeRawReplayRecipe(meas, meas.last_lio_update_time,
                                m.lio_time, true);
      if (raw_lidar_reprocess_enabled)
      {
        std::vector<RawLidarFrame> next_carry;
        for (const RawLidarFrame &frame : raw_lidar_carry_frames)
        {
          meas.raw_lidar_frames.push_back(frame);
          if (fast_livo::raw_replay::contributesAfter(frame, m.lio_time))
            next_carry.push_back(frame);
        }
        raw_lidar_carry_frames.swap(next_carry);
      }

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);
      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        if (raw_lidar_reprocess_enabled)
        {
          if (lid_raw_message_buffer.empty())
            throw std::runtime_error("raw LiDAR buffer missing LIVO scan");
          const RawLidarFrame &raw_frame = lid_raw_message_buffer.front();
          if (std::abs(raw_frame.effective_header_time - frame_header_time) >
              1e-9)
            throw std::runtime_error("raw/processed LiDAR buffer timestamp mismatch");
          meas.raw_lidar_frames.push_back(raw_frame);
          if (fast_livo::raw_replay::contributesAfter(raw_frame, m.lio_time))
            raw_lidar_carry_frames.push_back(raw_frame);
        }
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
        if (raw_lidar_reprocess_enabled) lid_raw_message_buffer.pop_front();
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO:
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = img_buffer.front();
      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = imu_buffer.front()->header.stamp.toSec();
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   imu_buffer.front()->header.stamp.toSec());
      // }
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false; // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      initializeRawReplayRecipe(meas, meas.lidar_frame_beg_time,
                                meas.lidar_frame_end_time, false);
      if (raw_lidar_reprocess_enabled)
      {
        if (lid_raw_message_buffer.empty())
          throw std::runtime_error("raw LiDAR buffer missing ONLY_LO scan");
        meas.raw_lidar_frames.push_back(lid_raw_message_buffer.front());
      }
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    if (raw_lidar_reprocess_enabled) lid_raw_message_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

// Provide output format for LiDAR-visual BA
void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  static int pub_num = 1;
  pub_num++;

  if (LidarMeasures.lio_vio_flg == VIO)
  {
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num >= pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo; pointRGB.b = pixel[0] * inv_expo;
          // if (pointRGB.r > 255) pointRGB.r = 255; else if (pointRGB.r < 0) pointRGB.r = 0;
          // if (pointRGB.g > 255) pointRGB.g = 255; else if (pointRGB.g < 0) pointRGB.g = 0;
          // if (pointRGB.b > 255) pointRGB.b = 255; else if (pointRGB.b < 0) pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (slam_mode_ == LIVO && LidarMeasures.lio_vio_flg == VIO)
  {
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  if (slam_mode_ == ONLY_LIO || slam_mode_ == ONLY_LO)
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); 
  }
  laserCloudmsg.header.stamp = ros::Time().fromSec(LidarMeasures.last_lio_update_time);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  double update_time = 0.0;
  if (LidarMeasures.lio_vio_flg == VIO) {
    update_time = LidarMeasures.measures.back().vio_time;
  } else { // LIO / LO
    update_time = LidarMeasures.measures.back().lio_time;
  }
  std::stringstream ss_time;
  ss_time << std::fixed << std::setprecision(6) << update_time;

  const bool persistent_output_allowed = !map_transaction_state ||
      !map_transaction_state->active() || transaction_replay_active;
  const bool replay_quarantine_content = persistent_output_allowed &&
      transaction_replay_active &&
      map_transaction_state->isCorruptInterval(update_time - _first_lidar_time);
  const bool map_content_allowed = persistent_output_allowed &&
      (!replay_quarantine_content || replay_pcd_inliers_in_quarantine);

  const bool committed_map_snapshot = !transaction_replay_active &&
      (!map_transaction_state || !map_transaction_state->active());
  if (shadow_online_map_publish_enabled && committed_map_snapshot &&
      std::isfinite(update_time) &&
      update_time - shadow_online_map_last_publish_time >=
          shadow_online_map_publish_period_sec)
  {
    PointCloudXYZI representative_map;
    voxelmap_manager->ExportRepresentativeCloud(
        representative_map, shadow_online_map_publish_max_points, 3);
    if (!representative_map.empty())
    {
      sensor_msgs::PointCloud2 map_message;
      pcl::toROSMsg(representative_map, map_message);
      map_message.header.stamp = ros::Time().fromSec(update_time);
      map_message.header.frame_id = "camera_init";
      pubLaserCloudMap.publish(map_message);
      shadow_online_map_last_publish_time = update_time;
    }
  }

  if (pcd_save_en && !bounded_voxel_export_en && map_content_allowed)
  {
    static int scan_wait_num = 0;

    switch (pcd_save_type)
    {
      case 0: /** world frame **/
        if (replay_quarantine_content && replay_pcd_inliers_in_quarantine)
        {
          // Do not reintroduce the complete faulted scan into the recovered
          // PCD.  Persist only the same residual-consistent correspondences
          // that were allowed to update the replay map.
          if ((LidarMeasures.lio_vio_flg == LIO ||
               LidarMeasures.lio_vio_flg == LO) && slam_mode_ == LIVO)
          {
            const double pcd_residual_limit = std::min(
                healthy_residual_limit,
                selective_lidar_gate_config.minimum_absolute_gate_m);
            for (const PointToPlane &inlier : voxelmap_manager->ptpl_list_)
            {
              if (inlier.source_index_ >= voxelmap_manager->pv_list_.size())
                continue;
              if (!std::isfinite(inlier.dis_to_plane_) ||
                  std::abs(inlier.dis_to_plane_) > pcd_residual_limit)
                continue;
              const pointWithVar &pv =
                  voxelmap_manager->pv_list_[inlier.source_index_];
              pcl::PointXYZRGB point;
              point.x = static_cast<float>(pv.point_w.x());
              point.y = static_cast<float>(pv.point_w.y());
              point.z = static_cast<float>(pv.point_w.z());
              point.r = point.g = point.b = 255;
              pcl_wait_save->push_back(point);
            }
          }
          else if (LidarMeasures.lio_vio_flg == LIO ||
                   LidarMeasures.lio_vio_flg == LO)
          {
            const double pcd_residual_limit = std::min(
                healthy_residual_limit,
                selective_lidar_gate_config.minimum_absolute_gate_m);
            for (const PointToPlane &inlier : voxelmap_manager->ptpl_list_)
            {
              if (inlier.source_index_ >= voxelmap_manager->pv_list_.size())
                continue;
              if (!std::isfinite(inlier.dis_to_plane_) ||
                  std::abs(inlier.dis_to_plane_) > pcd_residual_limit)
                continue;
              const pointWithVar &pv =
                  voxelmap_manager->pv_list_[inlier.source_index_];
              PointType point;
              point.x = static_cast<float>(pv.point_w.x());
              point.y = static_cast<float>(pv.point_w.y());
              point.z = static_cast<float>(pv.point_w.z());
              point.intensity = 0.0f;
              pcl_wait_save_intensity->push_back(point);
            }
          }
        }
        else if (slam_mode_ == LIVO)
        {
          *pcl_wait_save += *laserCloudWorldRGB;
        }
        else
        {
          *pcl_wait_save_intensity += *pcl_w_wait_pub;
        }
        if(LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO) scan_wait_num++;
        break;

      case 1: /** body frame **/
        if (LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO)
        {
          int size = feats_undistort->points.size();
          PointCloudXYZI::Ptr laserCloudBody(new PointCloudXYZI(size, 1));
          for (int i = 0; i < size; i++)
          {
            RGBpointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudBody->points[i]);
          }
          *pcl_wait_save_intensity += *laserCloudBody;
          scan_wait_num++;
          cout << "save body frame points: " << pcl_wait_save_intensity->points.size() << endl;
        }
        pcd_save_interval = 1;
        
        break;

      default:
        pcd_save_interval = 1;
        scan_wait_num++;
        break;
    }
    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      const std::string all_points_dir =
          joinPath(pcd_output_directory, ss_time.str() + std::string(".pcd"));

      pcl::PCDWriter pcd_writer;

      cout << "current scan saved to " << all_points_dir << endl;
      if (pcl_wait_save->points.size() > 0)
      {
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
        PointCloudXYZRGB().swap(*pcl_wait_save);
      }
      if(pcl_wait_save_intensity->points.size() > 0)
      {
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
        PointCloudXYZI().swap(*pcl_wait_save_intensity);
      }
      scan_wait_num = 0;
    }
    
    if(LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO)
    {
      Eigen::Quaterniond q(_state.rot_end);
      fout_lidar_pos << std::fixed << std::setprecision(6);
      fout_lidar_pos <<  LidarMeasures.measures.back().lio_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.x() << " " << q.y() << " " << q.z()
          << " " << q.w() << " " << endl;
    }
  }
  if (img_save_en && persistent_output_allowed && LidarMeasures.lio_vio_flg == VIO)
  {
    static int img_wait_num = 0;
    img_wait_num++;

    if (img_save_interval > 0 && img_wait_num >= img_save_interval)
    {
      imwrite(string(string(ROOT_DIR) + "Log/image/") + ss_time.str() + string(".png"), vio_manager->img_rgb);
      
      Eigen::Quaterniond q(_state.rot_end);
      fout_visual_pos << std::fixed << std::setprecision(6);
      fout_visual_pos << LidarMeasures.measures.back().vio_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
      img_wait_num = 0;
    }
  }

  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub); 
  if(LidarMeasures.lio_vio_flg == VIO)  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time().fromSec(LidarMeasures.last_lio_update_time);
  set_posestamp(odomAftMapped.pose.pose);
  std::fill(odomAftMapped.pose.covariance.begin(),
            odomAftMapped.pose.covariance.end(), 0.0);
  for (int row = 0; row < 3; ++row)
  {
    for (int column = 0; column < 3; ++column)
    {
      odomAftMapped.pose.covariance[row * 6 + column] =
          _state.cov(3 + row, 3 + column);
      odomAftMapped.pose.covariance[(row + 3) * 6 + column + 3] =
          _state.cov(row, column);
      odomAftMapped.pose.covariance[row * 6 + column + 3] =
          _state.cov(3 + row, column);
      odomAftMapped.pose.covariance[(row + 3) * 6 + column] =
          _state.cov(row, 3 + column);
    }
  }

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );
  pubOdomAftMapped.publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time().fromSec(LidarMeasures.last_lio_update_time);
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time().fromSec(LidarMeasures.last_lio_update_time);
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  path.header.stamp = msg_body_pose.header.stamp;
  pubPath.publish(path);
}

#include "nominal_replay_probe.inc"
#include "causal_replay_probe.inc"
