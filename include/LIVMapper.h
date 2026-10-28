/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "map_transaction_state.h"
#include "online_harm_detector.h"
#include "recovery_proposal_arbiter.h"
#include "recovery_proposal_corroboration.h"
#include "healthy_tail_validator.h"
#include "recovery_benefit_validator.h"
#include "replay_payload_journal.h"
#include "replay_range_scale_oracle.h"
#include "range_scale_correction_estimator.h"
#include "lidar_frozen_scan_profile.h"
#include "raw_lidar_reprocess.h"
#include "vio.h"
#include "preprocess.h"
#include "fusion_event_audit.h"
#include "trajectory_journal_flag.h"
#include <set>
#include "input_event_profile.h"
#include "p2_runtime_receipt.h"
#include "p2_live_metric_receipt.h"
#include "incremental_public_guard.h"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt64.h>
#include <vikit/camera_loader.h>

class LIVMapper
{
public:
  struct MapperCheckpoint;

  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it);
  void initializeComponents();
  void initializeFiles();
  void run();
  fast_livo::audit::FusionEventAudit fusion_audit;
  std::uint64_t fusion_audit_event = 0;
  fast_livo::p2_receipt::Writer p2_runtime_receipt;
  fast_livo::p2_live_metric::Writer p2_live_metric_receipt;
  fast_livo::p2_live_metric::Metric p2_live_metric_cache;
  std::string p2_live_metric_route;
  std::uint64_t p2_live_metric_event_id = 0;
  std::uint64_t p2_live_metric_consumed_ns = 0;
  void recordP2Fusion(const char *record_type, const char *route,
                      std::uint64_t event_id, const char *stage,
                      const LidarMeasureGroup &group);
  void recordP2Proposal(std::uint64_t episode_id,
                        std::uint64_t proposal_index_in_episode,
                        bool frozen_forced, double onset_lower_sec,
                        double onset_point_sec, double onset_upper_sec);
  std::string p2_current_route;
  std::uint64_t p2_current_event_id = 0;
  bool p2_current_fusion_pending = false;
  std::string p2_current_modality;
  std::uint64_t p2_current_sensor_max_input_stamp_ns = 0;
  std::uint64_t p2_active_episode_id = 0;
  std::uint64_t p2_active_proposal_index = 0;
  std::string p2_active_proposal_origin;
  std::uint64_t p2_replay_begin_ns = 0;
  std::uint64_t p2_replay_end_ns = 0;
  std::uint64_t p2_replay_consumed_events = 0;
  std::uint64_t p2_replay_total_events = 0;
  std::uint64_t p2_rearm_count = 0;
  fast_livo::audit::InputEventProfile input_event_profile;
  std::uint64_t input_profile_lidar_sequence = 0;
  std::uint64_t input_profile_imu_sequence = 0;
  std::uint64_t input_profile_image_sequence = 0;
  std::uint64_t rolling_history_profile_sequence = 0;
  bool diagnostic_forbid_recovery = false;
  bool vio_patch_boundary_check_only = false;
  bool vio_patch_boundary_guard = false;
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  bool writeSourceRevisionFreezeReceipt(const std::string &raw_map_path,
                                        std::size_t map_point_count);
  void sourceRevisionCutRequestCallback(const std_msgs::UInt64::ConstPtr &message);
  void maybeWriteSourceRevisionCut();
  void exportRejectedCandidateMapEvidence(const std::string &reason,
                                          double relative_time_sec);
  void processImu();
  void beginMapTransaction(double relative_time);
  bool bufferMapTransactionEvent(double relative_time);
  void replayMapTransaction();
  void rollbackMapTransaction(const std::string &reason);
  void rejectCandidateAndRestoreIncumbent(const std::string &reason);
  void restoreMapTransactionCheckpoint();
  void restoreMapperCheckpoint(const MapperCheckpoint &checkpoint);
  MapperCheckpoint captureMapperCheckpoint() const;
  static std::vector<std::uint8_t> serializeMapperCheckpoint(
      const MapperCheckpoint &checkpoint);
  static MapperCheckpoint deserializeMapperCheckpoint(
      const std::vector<std::uint8_t> &payload);
  static std::uint64_t mapperCheckpointFingerprint(
      const MapperCheckpoint &checkpoint);
  void initializeTransactionDiagnostics();
  void logTransactionDiagnostic(const std::string &event, double relative_time,
                                const StatesGroup *reference = nullptr,
                                long long buffer_index = -1,
                                int quarantined = -1);
  void maintainRollingHistory(double relative_time);
  bool activateRollingTransaction(double relative_time);
  void saveTrajectoryPoses();
  void publishTransactionStatus(const std::string &state,
                                 const std::string &detail = std::string(),
                                 double relative_time_sec =
                                     std::numeric_limits<double>::quiet_NaN());
  void shadowAbortCallback(const std_msgs::Bool::ConstPtr &message);
  void recoveryProposalCorroborationCallback(
      const std_msgs::String::ConstPtr &message);
  void processPendingExternalRecoveryProposal(double relative_time_sec);
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
  void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po);
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void annotateOracleRangeScaleProvenance(PointCloudXYZI::Ptr &cloud,
                                          double sensor_header_time_sec);
  void initializeRawReplayRecipe(LidarMeasureGroup &group,
                                 double begin_time, double end_time,
                                 bool split_by_time);
  bool rebuildProcessedCloudFromRaw(LidarMeasureGroup &group,
                                    double event_relative_time,
                                    std::size_t &corrected_points,
                                    double diagnostic_inverse_scale =
                                        std::numeric_limits<double>::quiet_NaN());
  void runPrivateFrozenScaleProbe(
      double relative_time, const StatesGroup &state_before_imu,
      const ImuProcess::Checkpoint &imu_before_event);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);
  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_visual_sub_map(const ros::Publisher &pubSubVisualMap);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_path(const ros::Publisher pubPath);
  void readParameters(ros::NodeHandle &nh);
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

  struct BufferedTransactionEvent
  {
    std::size_t sequence = 0;
    double relative_time = 0.0;
    LidarMeasureGroup measures;
    fast_livo::replay_storage::Record payload_record;
    double incumbent_raw_residual = std::numeric_limits<double>::quiet_NaN();
    double incumbent_raw_match_ratio = std::numeric_limits<double>::quiet_NaN();
    double incumbent_normalized_innovation =
        std::numeric_limits<double>::quiet_NaN();
    bool incumbent_health_valid = false;
    StatesGroup incumbent_state;
    bool incumbent_state_valid = false;
  };

  struct RecoveryReplayCoverage
  {
    bool continuous = false;
    double first_event_relative_time =
        std::numeric_limits<double>::quiet_NaN();
    double gap_sec = std::numeric_limits<double>::infinity();
  };

  static RecoveryReplayCoverage assessRecoveryReplayCoverage(
      double checkpoint_relative_time,
      const std::deque<BufferedTransactionEvent> &events,
      double maximum_gap_sec);

  struct MapperCheckpoint
  {
    StatesGroup state;
    StatesGroup propagated_state;
    StatesGroup latest_ekf_state;
    StatesGroup imu_propagate;
    ImuProcess::Checkpoint imu_process;
    std::vector<pointWithVar> point_variances;
    PointCloudXYZI lidar_world_for_visual;
    PointCloudXYZI pending_publish_cloud;
    V3D voxel_position_last;
    V3D voxel_last_slide_position;
    double latest_ekf_time = 0.0;
    double last_completed_event_relative_time = -1.0;
    double average_processing_time = 0.0;
    std::size_t path_pose_count = 0;
    std::size_t trajectory_pose_count = 0;
    std::size_t pcd_rgb_point_count = 0;
    std::size_t pcd_intensity_point_count = 0;
    int voxel_frame_id = 0;
    int voxel_next_plane_id = -1; // -1: legacy checkpoint lacks this field.
    int visual_frame_counter = 0;
    int visual_processed_frames = 0;
    int processed_lidar_frames = 0;
    bool lidar_map_inited = false;
    bool ekf_finish_once = false;
    bool state_update_flg = false;
  };

  struct RollingCheckpoint
  {
    double relative_time = 0.0;
    std::size_t first_event_sequence = 0;
    std::size_t voxel_count = 0;
    std::size_t voxel_journal_id = 0;
    std::size_t visual_journal_id = 0;
    bool uses_delta_voxel_journal = false;
    bool uses_delta_visual_journal = false;
    MapperCheckpoint mapper;
    std::unique_ptr<VoxelMapSnapshot> voxel_map;
    std::shared_ptr<VIOCheckpoint> visual_map;
    double mapper_copy_wall_sec = 0.0;
    double voxel_copy_wall_sec = 0.0;
    double visual_copy_wall_sec = 0.0;
    double total_copy_wall_sec = 0.0;
  };

  // Exact incumbent state at the live replay boundary.  The map components
  // are dirty-set after-images; append-only outputs retain only the tail after
  // the selected rollback anchor.  Rejection can therefore return to the
  // public incumbent without re-running estimator events.
  struct IncumbentBoundaryImage
  {
    bool valid = false;
    double relative_time = -1.0;
    std::size_t next_event_sequence = 0;
    MapperCheckpoint mapper;
    std::shared_ptr<VIOCheckpoint> visual_metadata;
    std::unique_ptr<VoxelMapManager::MutationRedoImage> voxel_redo;
    std::unique_ptr<VIOManager::MutationRedoImage> visual_redo;
    std::vector<geometry_msgs::PoseStamped> path_tail;
    std::vector<std::pair<double, std::array<double, 7>>> trajectory_tail;
    PointCloudXYZRGB pcd_rgb_tail;
    PointCloudXYZI pcd_intensity_tail;
    double capture_wall_sec = 0.0;
  };

  // Self-contained, versioned seed transferred from the public incumbent to
  // an isolated recovery process. Component payloads have their own checksums;
  // the outer envelope additionally protects boundary metadata and replay
  // offsets so a partially published generation can never be admitted.
  struct RecoveryHandoffBundle
  {
    // Diagnostic binding only; not part of the serialized handoff format.
    std::string receipt_file_sha256;
    std::uint64_t generation = 0;
    double boundary_relative_time = -1.0;
    std::size_t next_event_sequence = 0;
    int visual_patch_size_total = 0;
    MapperCheckpoint mapper;
    std::unique_ptr<VoxelMapSnapshot> voxel_map;
    std::shared_ptr<VIOCheckpoint> visual_map;
    fast_livo::harm_detection::OnlineHarmDetector::Snapshot detector;
    std::string replay_journal_path;
    std::vector<fast_livo::replay_storage::Record> replay_records;
  };

  struct RecoveryHandoffHeader
  {
    std::uint64_t generation = 0;
    double boundary_relative_time = -1.0;
    std::size_t next_event_sequence = 0;
    int visual_patch_size_total = 0;
  };

  std::string nominal_probe_directory;
  double nominal_probe_start = 10.0, nominal_probe_end = 12.0;
  std::unique_ptr<RecoveryHandoffBundle> nominal_probe_seed;
  std::vector<std::pair<std::uint64_t, LidarMeasureGroup>> nominal_probe_events;
  void nominalProbeBeforeEvent(double relative_time);
  bool nominalProbeAfterEvent(double relative_time);
  RecoveryHandoffBundle captureNominalProbeBundle();
  LidarMeasureGroup cloneNominalProbeEvent(const LidarMeasureGroup &source) const;
  // Development-only causal probe. Never enabled by a production preset.
  std::string causal_event_archive_directory, causal_reference_event_directory;
  std::string causal_cloud_audit_directory;
  double causal_archive_start = 78.0, causal_archive_end = 95.0;
  void causalArchiveDirectEvent(double relative_time);
  void causalReplaceReplayCloud(double relative_time);
  void causalDumpCloud(const std::string &route, const std::string &stage,
                       double relative_time, const PointCloudXYZI::Ptr &cloud) const;

  static std::vector<std::uint8_t> serializeRecoveryHandoffBundle(
      const RecoveryHandoffBundle &bundle);
  static RecoveryHandoffBundle deserializeRecoveryHandoffBundle(
      const std::vector<std::uint8_t> &payload);
  static RecoveryHandoffBundle deserializeRecoveryHandoffBundle(
      const std::uint8_t *payload, std::size_t payload_size);
  static void writeRecoveryHandoffBundleAtomic(
      const std::string &path, const RecoveryHandoffBundle &bundle);
  static RecoveryHandoffHeader readRecoveryHandoffHeader(
      const std::string &path);
  static RecoveryHandoffBundle readRecoveryHandoffBundle(
      const std::string &path);
  // Empty string means the sealed journal may be shared. Any other string is
  // a fail-closed reason. This does not open the journal for append.
  static std::string authenticatePrivateForkJournalFiles(
      const std::string &bundle_journal_path,
      const std::string &sealed_journal_path,
      const std::string &writable_spill_path);
  void maybeExportRecoveryHandoff(double incoming_relative_time);
  void markRecoveryHandoffConsumed(std::uint64_t generation,
                                   double boundary_relative_time);
  std::unique_ptr<RecoveryHandoffBundle> loadRecoveryHandoffBefore(
      double onset_relative_time) const;

  LidarMeasureGroup cloneMeasureGroup(const LidarMeasureGroup &source) const;
  void resetReplayPayloadJournal();
  void spillReplayPayload(BufferedTransactionEvent &event);
  LidarMeasureGroup materializeReplayEvent(const BufferedTransactionEvent &event);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name, img_topic;
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, bounded_voxel_export_en = false, img_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false;
  int img_save_interval = 1, pcd_save_interval = -1, pcd_save_type = 0;
  std::size_t bounded_voxel_export_max_points = 750000;
  std::size_t bounded_voxel_export_samples_per_plane = 9;
  int pub_scan_num = 1;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  // Relative timestamp of the last event whose mapping update completed.
  // Rolling checkpoints are captured before the incoming event is processed,
  // so their boundary label must come from this value rather than sync state.
  double last_completed_event_relative_time = -1.0;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  double outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<RawLidarFrame> lid_raw_message_buffer;
  std::vector<RawLidarFrame> raw_lidar_carry_frames;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  deque<cv::Mat> img_buffer;
  deque<double> img_time_buffer;
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  double IMG_POINT_COV;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pre, fout_out, fout_visual_pos, fout_lidar_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_img;
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubPath;
  ros::Publisher transaction_status_pub;
  ros::Subscriber shadow_abort_sub;
  ros::Subscriber recovery_proposal_corroboration_sub;
  ros::Subscriber source_revision_cut_request_sub;
  ros::Publisher source_revision_cut_ready_pub;
  bool source_revision_cut_enabled = false;
  bool source_revision_payload_hash_enabled = false;
  bool source_revision_barrier_enabled = false;
  bool source_revision_cut_pending = false;
  bool source_revision_cut_attempted = false;
  std::uint64_t source_revision_cut_switch_stamp_ns = 0;
  double source_revision_cut_request_wall_sec = 0.0;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  image_transport::Publisher pubImage;
  ros::Publisher mavros_pose_publisher;
  ros::Timer imu_prop_timer;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;

  fast_livo::transactional_mapping::Config map_transaction_config;
  std::unique_ptr<fast_livo::transactional_mapping::StateMachine> map_transaction_state;
  MapperCheckpoint map_transaction_checkpoint;
  std::unique_ptr<VoxelMapSnapshot> voxel_map_checkpoint;
  std::shared_ptr<VIOCheckpoint> vio_map_checkpoint;
  std::deque<BufferedTransactionEvent> map_transaction_buffer;
  std::deque<BufferedTransactionEvent> rolling_event_buffer;
  std::deque<RollingCheckpoint> rolling_checkpoints;
  std::map<double, std::array<double, 7>> trajectory_poses;
  std::size_t next_transaction_event_sequence = 0;
  double transaction_event_clone_total_wall_sec = 0.0;
  double transaction_event_clone_maximum_wall_sec = 0.0;
  std::size_t transaction_event_clone_count = 0;
  std::size_t peak_transaction_buffer_events = 0;
  double rolling_event_clone_total_wall_sec = 0.0;
  double rolling_event_clone_maximum_wall_sec = 0.0;
  std::size_t rolling_event_clone_count = 0;
  std::size_t peak_rolling_event_buffer_events = 0;
  double selected_checkpoint_mapper_copy_wall_sec = 0.0;
  double selected_checkpoint_voxel_copy_wall_sec = 0.0;
  double selected_checkpoint_visual_copy_wall_sec = 0.0;
  double selected_checkpoint_total_copy_wall_sec = 0.0;
  double selected_checkpoint_relative_time_sec = -1.0;
  std::size_t transaction_checkpoint_voxel_count = 0;
  std::size_t transaction_voxel_journal_id = 0;
  std::size_t transaction_visual_journal_id = 0;
  std::unique_ptr<IncumbentBoundaryImage> replay_incumbent_boundary;
  bool delta_voxel_journal_enabled = false;
  bool transaction_uses_delta_voxel_journal = false;
  bool transaction_uses_delta_visual_journal = false;
  bool rolling_checkpoint_rotation_safe = true;
  bool transaction_replay_active = false;
  bool lidar_updates_enabled = true;
  bool visual_updates_enabled = true;
  bool lidar_map_writes_enabled = true;
  bool visual_map_writes_enabled = true;
  bool freeze_vio_provisional = true;
  bool live_lidar_provisional_writes = true;
  bool replay_lidar_updates_in_quarantine = false;
  bool replay_visual_updates_in_quarantine = false;
  bool replay_map_writes_in_quarantine = false;
  bool replay_map_writes_during_validation_tail = true;
  bool selective_lidar_gate_during_validation_tail = false;
  bool selective_map_write_gate_during_validation_tail = false;
  bool selective_map_write_gate_active = false;
  bool replay_reanchor_pose_to_incumbent_at_commit = false;
  bool replay_output_boundary_alignment_enabled = false;
  bool replay_pcd_inliers_in_quarantine = false;
  double replay_fault_lidar_weight = 1.0;
  fast_livo::replay_oracle::RangeScaleConfig replay_range_scale_oracle_config;
  fast_livo::replay_oracle::RangeScaleConfig private_raw_scan_oracle_config;
  double replay_range_scale_oracle_epoch_sensor_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  bool estimated_range_scale_candidate_enabled = false;
  double estimated_range_scale_candidate_terminal_scale = 1.0;
  double estimated_range_scale_minimum_onset_confidence = 0.60;
  double estimated_range_scale_post_detection_horizon_sec = 8.0;
  double estimated_range_scale_candidate_ramp_end_offset_sec = 0.0;
  std::string estimated_range_scale_candidate_ramp_start_anchor = "lower";
  fast_livo::correction_estimation::CorrectionPlan
      estimated_range_scale_candidate_plan;
  // Private frozen input operator; never a public admission mechanism.
  bool private_lidar_profile_enabled = false;
  std::string private_lidar_profile_path;
  fast_livo::scan_profile::FrozenProfile private_lidar_profile;
  void preflightPrivateLidarProfile();
  double raw_lidar_reprocess_epoch_sensor_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  double mapper_first_lidar_header_sec =
      std::numeric_limits<double>::quiet_NaN();
  double private_fork_sealed_time_origin_sec =
      std::numeric_limits<double>::quiet_NaN();
  double private_fork_completed_event_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  int private_fork_next_event_type = 0;
  double private_fork_next_event_absolute_time_sec =
      std::numeric_limits<double>::quiet_NaN();
  std::string private_fork_next_event_cur_hash;
  std::string private_fork_next_event_image_hash;
  std::string private_fork_next_event_imu_hash;
  int private_fork_next_event_cur_count = 0;
  bool raw_lidar_reprocess_enabled = false;
  PreprocessPtr replay_preprocessor;
  std::string raw_lidar_reprocess_audit_path;
  std::ofstream raw_lidar_reprocess_audit;
  std::size_t raw_lidar_reprocess_events = 0;
  std::size_t raw_lidar_reprocess_frames = 0;
  std::size_t raw_lidar_reprocess_bytes = 0;
  std::size_t raw_lidar_reprocess_corrected_points = 0;
  std::size_t raw_lidar_reprocess_output_points = 0;
  std::size_t raw_lidar_reprocess_failures = 0;
  bool selective_lidar_gate_enabled = false;
  fast_livo::selective_gating::Config selective_lidar_gate_config;
  bool robust_lidar_weight_enabled = false;
  fast_livo::robust_weighting::Config robust_lidar_weight_config;
  std::string transaction_diagnostics_path;
  std::ofstream transaction_diagnostics;
  std::string harm_detector_diagnostics_path;
  std::ofstream harm_detector_diagnostics;
  std::string trajectory_journal_path;
  std::ofstream trajectory_journal;
  bool disk_backed_replay_payload_enabled = false;
  bool async_replay_payload_spill_enabled = false;
  std::uint64_t async_replay_payload_queue_bytes = 67108864;
  std::string replay_payload_journal_path;
  fast_livo::replay_storage::Journal replay_payload_journal;
  std::string shadow_abort_topic;
  bool shadow_rearm_after_internal_rejection = false;
  bool rejected_candidate_map_export_enabled = false;
  bool private_candidate_map_receipt_enabled = false;
  std::string private_candidate_map_receipt_directory;
  std::string private_candidate_checkpoint_sha256;
  std::uint64_t private_candidate_handoff_generation = 0;
  void exportPrivateCandidateMapReceipt(double replay_boundary_time);
  void maybeExportTypedEventMapSnapshot(const char *route,
                                        std::uint64_t event_sequence);
  bool typed_event_map_snapshot_enabled = false;
  std::string typed_event_map_snapshot_directory;
  std::string typed_event_map_snapshot_event_type;
  double typed_event_map_snapshot_relative_time =
      std::numeric_limits<double>::quiet_NaN();
  std::string typed_event_map_snapshot_branch_role;
  double typed_event_map_snapshot_checkpoint_completed =
      std::numeric_limits<double>::quiet_NaN();
  std::string typed_event_map_snapshot_journal_sha256;
  bool typed_event_map_snapshot_written = false;
  bool typed_event_snapshot_closed = false;
  bool incumbent_rebuild_trajectory_active = false;
  std::set<std::string> production_replay_trajectory_stamps;
  std::string rejected_candidate_map_export_directory;
  std::size_t rejected_candidate_map_export_max_points = 750000;
  std::size_t rejected_candidate_map_export_samples_per_plane = 9;
  std::string pcd_output_directory;
  // Default-off representative map stream for public-mux continuity audit.
  bool shadow_online_map_publish_enabled = false;
  double shadow_online_map_publish_period_sec = 10.0;
  std::size_t shadow_online_map_publish_max_points = 25000;
  double shadow_online_map_last_publish_time =
      -std::numeric_limits<double>::infinity();
  StatesGroup replay_incumbent_state;
  bool replay_incumbent_state_valid = false;
  bool rolling_history_enabled = false;
  double rolling_checkpoint_period_sec = 5.0;
  double rolling_history_horizon_sec = 20.0;
  std::size_t rolling_max_checkpoints = 5;
  std::size_t rolling_delta_max_checkpoints = 3;
  bool rolling_preserve_horizon_anchor = false;
  double recovery_handoff_maximum_replay_gap_sec = 0.25;
  std::string recovery_handoff_export_directory;
  std::string recovery_handoff_import_directory;
  std::uint64_t recovery_handoff_generation = 0;
  double last_recovery_handoff_export_time = -1.0;
  std::deque<std::string> recovery_handoff_export_files;
  bool recovery_handoff_stop_export_after_consume = false;
  bool recovery_handoff_export_consumed = false;
  // Default off. Source export records one journal path inside the bundle;
  // importers require that same file to be read-only and distinct from the
  // consumer spill journal. Existing source-import mode leaves both false.
  bool private_fork_bind_journal = false;
  bool private_fork_require_readonly_journal = false;
  std::string private_fork_sealed_journal_path;
  fast_livo::harm_detection::Config harm_detector_config;
  std::unique_ptr<fast_livo::harm_detection::OnlineHarmDetector> harm_detector;
  fast_livo::recovery_scheduling::Config recovery_proposal_arbiter_config;
  std::unique_ptr<fast_livo::recovery_scheduling::RecoveryProposalArbiter>
      recovery_proposal_arbiter;
  std::string recovery_proposal_corroboration_topic;
  double recovery_proposal_corroboration_tolerance_sec = 12.0;
  struct PendingExternalRecoveryProposal
  {
    bool valid = false;
    bool corroboration_ready = false;
    std::size_t episode_id = 0;
    std::size_t proposal_index_in_episode = 0;
    double detection_sec = -1.0;
    double point_onset_sec = -1.0;
    double onset_lower_bound_sec = -1.0;
    double onset_upper_bound_sec = -1.0;
    double residual_score = 0.0;
    double match_score = 0.0;
    double innovation_score = 0.0;
    double raw_residual = 0.0;
    double healthy_residual_limit = 0.0;
    double healthy_match_ratio_limit = 0.0;
    double healthy_match_baseline_reference = 0.0;
    bool healthy_validator_limits_ready = false;
    double protected_checkpoint_time_sec = -1.0;
    double corroboration_detection_sec = -1.0;
    std::string proposal_detail;
  };
  std::mutex recovery_proposal_corroboration_mutex;
  fast_livo::recovery_scheduling::ProposalCorroboration
      latest_supported_corroboration;
  PendingExternalRecoveryProposal pending_external_recovery_proposal;
  fast_livo::recovery_validation::Config healthy_tail_config;
  fast_livo::recovery_validation::BenefitConfig recovery_benefit_config;
  bool recovery_benefit_allow_absolute_health_dominance = false;
  bool recovery_benefit_common_reference_map_enabled = false;
  bool healthy_tail_commit_dry_run = false;
  bool online_harm_detector_enabled = false;
  bool harm_detector_observe_only = false;
  fast_livo::incremental_public_guard::Guard incremental_public_guard;
  fast_livo::incremental_public_guard::EvidenceTracker incremental_guard_evidence;
  fast_livo::incremental_public_guard::Config incremental_guard_config;
  bool private_frozen_scale_probe_enabled = false;
  double private_frozen_scale_probe_reference_sec = 45.0;
  double private_frozen_scale_probe_period_sec = 0.7;
  double private_frozen_scale_probe_end_sec = 65.0;
  double private_frozen_scale_probe_last_sec = -1.0;
  bool private_frozen_scale_probe_map_frozen = false;
  bool private_frozen_scale_probe_snapshot_reference_enabled = false;
  bool private_frozen_scale_probe_pose_refinement_enabled = false;
  bool private_frozen_scale_probe_common_support_enabled = false;
  std::unique_ptr<VoxelMapSnapshot> private_frozen_scale_probe_reference_snapshot;
  std::ofstream private_frozen_scale_probe_csv;
  struct FrozenDetectionEvent
  {
    bool enabled = false;
    bool fired = false;
    bool missed = false;
    double detection_sec = -1.0;
    double lower_sec = -1.0;
    double point_sec = -1.0;
    double upper_sec = -1.0;
    double confidence = -1.0;
    std::size_t episode_id = 0;
    std::size_t proposal_index_in_episode = 0;
    std::string source_manifest_sha256;
  } frozen_detection_event;
  bool online_detection_pending = false;
  double online_point_onset_sec = -1.0;
  double online_detected_onset_sec = -1.0;
  double online_detection_time_sec = -1.0;
  double detector_normalized_innovation = 0.0;
  double protected_rolling_checkpoint_time_sec = -1.0;
  double online_quarantine_duration_sec = 8.0;
  double healthy_residual_mad_multiplier = 3.0;
  double healthy_residual_increase_floor = 0.0015;
  double healthy_match_mad_multiplier = 3.0;
  double healthy_match_drop_floor = 0.05;
  double healthy_residual_limit = 0.0;
  double healthy_match_ratio_limit = 0.0;
  double healthy_match_baseline_reference = 0.0;
  bool healthy_validator_limits_ready = false;
  std::deque<double> healthy_residual_history;
  std::deque<double> healthy_match_history;
  std::size_t healthy_baseline_window = 100;
};
#endif
