#include "shadow_generation_selector.h"
#include "shadow_frame_epoch_contract.h"
#include "source_revision_hash.h"

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt64.h>
#include <tf/transform_broadcaster.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>

namespace sm = fast_livo::shadow_mux;

class ShadowOutputMux
{
public:
  ShadowOutputMux() : private_nh_("~"), selector_(loadConfig())
  {
    private_nh_.param<bool>("frame_epoch_diagnostics_enabled",
                            frame_epoch_diagnostics_enabled_, false);
    private_nh_.param<bool>("frame_epoch_snapshot_preview_enabled",
                            frame_epoch_snapshot_preview_enabled_, false);
    const char *cut_env = std::getenv("SHADOW_SOURCE_REVISION_CUT_ENABLED");
    const char *barrier_env = std::getenv("SHADOW_SOURCE_REVISION_BARRIER_ENABLED");
    source_revision_barrier_enabled_ =
        barrier_env != nullptr && std::string(barrier_env) == "true";
    const char *hash_env = std::getenv("SHADOW_SOURCE_REVISION_PAYLOAD_HASH_ENABLED");
    source_revision_payload_hash_enabled_ =
        hash_env != nullptr && std::string(hash_env) == "true";
    if (source_revision_payload_hash_enabled_ && !source_revision_barrier_enabled_)
      throw std::invalid_argument(
          "source revision payload hash requires prepublic barrier");
    source_revision_cut_request_enabled_ =
        source_revision_barrier_enabled_ ||
        (cut_env != nullptr && std::string(cut_env) == "true");
    if (source_revision_barrier_enabled_)
    {
      private_nh_.param<double>("source_revision_barrier_timeout_sec",
                                source_revision_barrier_timeout_sec_,
                                source_revision_payload_hash_enabled_ ? 10.0 : 2.0);
      if (!std::isfinite(source_revision_barrier_timeout_sec_) ||
          source_revision_barrier_timeout_sec_ < 0.1 ||
          source_revision_barrier_timeout_sec_ > 10.0 ||
          !selector_.positionBridgeEnabled())
        throw std::invalid_argument(
            "source revision barrier requires position bridge and timeout in [0.1,10] s");
    }
    if (frame_epoch_diagnostics_enabled_ && !selector_.positionBridgeEnabled())
      throw std::invalid_argument(
          "frame_epoch_diagnostics_enabled requires correction-aware position bridge");
    if (frame_epoch_snapshot_preview_enabled_ && !frame_epoch_diagnostics_enabled_)
      throw std::invalid_argument(
          "frame_epoch_snapshot_preview_enabled requires frame_epoch_diagnostics_enabled");
    private_nh_.param<double>("incumbent_cache_horizon_sec",
                              cache_horizon_sec_, 180.0);
    if (cache_horizon_sec_ <= 0.0)
      throw std::invalid_argument("incumbent_cache_horizon_sec must be positive");
    private_nh_.param<std::string>("boundary_diagnostics_path",
                                   boundary_diagnostics_path_, std::string());
    private_nh_.param<std::string>("callback_trace_path",
                                   callback_trace_path_, std::string());
    if (!callback_trace_path_.empty())
    {
      callback_trace_.open(callback_trace_path_, std::ios::out | std::ios::trunc);
      if (!callback_trace_)
        throw std::runtime_error("cannot open callback_trace_path: " +
                                 callback_trace_path_);
      callback_trace_ << "callback_index,receipt_wall_sec,event,sensor_time_sec,"
                         "selector_state_before,latest_published_incumbent_time_sec,"
                         "last_active_odom_time_sec,published_odom_watermark_sec,"
                         "frame_id,child_frame_id,x,y,z,qx,qy,qz,qw,"
                         "pose_covariance_row_major,status\n";
      callback_trace_.flush();
    }
    if (!boundary_diagnostics_path_.empty())
    {
      boundary_diagnostics_.open(boundary_diagnostics_path_,
                                 std::ios::out | std::ios::trunc);
      if (!boundary_diagnostics_)
        throw std::runtime_error("cannot open boundary_diagnostics_path: " +
                                 boundary_diagnostics_path_);
      boundary_diagnostics_
          << "event,receipt_wall_sec,status,status_sensor_time_sec,"
             "status_relative_time_sec,candidate_callback_sequence,"
             "latest_candidate_callback_time_sec,maximum_candidate_time_sec,"
             "latest_published_incumbent_time_sec,candidate_time_sec,"
             "incumbent_time_sec,signed_match_dt_sec,absolute_match_dt_sec,"
             "candidate_minus_status_sensor_sec,"
             "candidate_lag_to_latest_incumbent_sec,within_timestamp_tolerance,"
             "position_divergence_m,rotation_divergence_deg,"
             "incumbent_x,incumbent_y,incumbent_z,candidate_x,candidate_y,"
             "candidate_z\n";
      boundary_diagnostics_.flush();
    }

    active_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 20);
    active_path_pub_ = nh_.advertise<nav_msgs::Path>("/path", 5, true);
    active_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 10);
    active_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/Laser_map", 2, true);
    generation_pub_ = private_nh_.advertise<std_msgs::String>("active_generation", 1, true);
    abort_pub_ = private_nh_.advertise<std_msgs::Bool>("candidate_abort", 1, true);
    if (source_revision_cut_request_enabled_)
      source_revision_cut_request_pub_ = nh_.advertise<std_msgs::UInt64>(
          "/shadow/candidate/source_revision_cut_request", 1, true);
    if (source_revision_barrier_enabled_)
    {
      source_revision_ready_sub_ = nh_.subscribe(
          "/shadow/candidate/source_revision_cut_ready", 1,
          &ShadowOutputMux::sourceRevisionReady, this);
      source_revision_manifest_pub_ = private_nh_.advertise<std_msgs::String>(
          "source_revision/manifest", 1, true);
      source_revision_barrier_timer_ = nh_.createWallTimer(
          ros::WallDuration(0.05), &ShadowOutputMux::sourceRevisionBarrierTimer,
          this);
    }
    if (frame_epoch_diagnostics_enabled_)
    {
      native_odom_pub_ = private_nh_.advertise<nav_msgs::Odometry>(
          "frame_epoch/candidate_native/odometry", 20);
      native_path_pub_ = private_nh_.advertise<nav_msgs::Path>(
          "frame_epoch/candidate_native/path", 5, true);
      native_cloud_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
          "frame_epoch/candidate_native/cloud", 10);
      native_map_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
          "frame_epoch/candidate_native/map", 2, true);
      frame_epoch_contract_pub_ = private_nh_.advertise<std_msgs::String>(
          "frame_epoch/contract", 1, true);
      if (frame_epoch_snapshot_preview_enabled_)
      {
        frame_epoch_versioned_manifest_pub_ = private_nh_.advertise<std_msgs::String>(
            "frame_epoch/versioned_manifest", 1, true);
        snapshot_preview_path_pub_ = private_nh_.advertise<nav_msgs::Path>(
            "frame_epoch/snapshot_preview/path", 2, true);
        snapshot_preview_map_pub_ = private_nh_.advertise<sensor_msgs::PointCloud2>(
            "frame_epoch/snapshot_preview/map", 2, true);
        snapshot_preview_bundle_pub_ = private_nh_.advertise<std_msgs::String>(
            "frame_epoch/snapshot_preview/bundle", 2, true);
      }
    }

    incumbent_odom_sub_ = nh_.subscribe("/shadow/incumbent/aft_mapped_to_init", 500,
        &ShadowOutputMux::incumbentOdom, this);
    candidate_odom_sub_ = nh_.subscribe("/shadow/candidate/aft_mapped_to_init", 500,
        &ShadowOutputMux::candidateOdom, this);
    incumbent_path_sub_ = nh_.subscribe("/shadow/incumbent/path", 20,
        &ShadowOutputMux::incumbentPath, this);
    candidate_path_sub_ = nh_.subscribe("/shadow/candidate/path", 20,
        &ShadowOutputMux::candidatePath, this);
    incumbent_cloud_sub_ = nh_.subscribe("/shadow/incumbent/cloud_registered", 20,
        &ShadowOutputMux::incumbentCloud, this);
    candidate_cloud_sub_ = nh_.subscribe("/shadow/candidate/cloud_registered", 20,
        &ShadowOutputMux::candidateCloud, this);
    incumbent_map_sub_ = nh_.subscribe("/shadow/incumbent/Laser_map", 5,
        &ShadowOutputMux::incumbentMap, this);
    candidate_map_sub_ = nh_.subscribe("/shadow/candidate/Laser_map", 5,
        &ShadowOutputMux::candidateMap, this);
    candidate_status_sub_ = nh_.subscribe("/shadow/candidate/transaction_status", 20,
        &ShadowOutputMux::candidateStatus, this);
    finalize_sub_ = private_nh_.subscribe("finalize", 1,
        &ShadowOutputMux::finalize, this);

    publishEvaluation();
  }

private:
  sm::Config loadConfig()
  {
    sm::Config config;
    int minimum_samples = static_cast<int>(config.minimum_matched_samples);
    int minimum_post_catchup_samples =
        static_cast<int>(config.minimum_post_catchup_samples);
    int minimum_consecutive_uncertainty_exceedance_samples = static_cast<int>(
        config.minimum_consecutive_uncertainty_exceedance_samples);
    private_nh_.param<double>("probation_duration_sec", config.probation_duration_sec,
                              config.probation_duration_sec);
    private_nh_.param<double>("timestamp_tolerance_sec", config.timestamp_tolerance_sec,
                              config.timestamp_tolerance_sec);
    private_nh_.param<double>("maximum_boundary_position_m",
                              config.maximum_boundary_position_m,
                              config.maximum_boundary_position_m);
    private_nh_.param<double>("maximum_boundary_rotation_deg",
                              config.maximum_boundary_rotation_deg,
                              config.maximum_boundary_rotation_deg);
    private_nh_.param<double>("maximum_position_growth_m",
                              config.maximum_position_growth_m,
                              config.maximum_position_growth_m);
    private_nh_.param<double>("maximum_rotation_growth_deg",
                              config.maximum_rotation_growth_deg,
                              config.maximum_rotation_growth_deg);
    private_nh_.param<int>("minimum_matched_samples", minimum_samples,
                           minimum_samples);
    config.minimum_matched_samples = static_cast<std::size_t>(std::max(1, minimum_samples));
    private_nh_.param<double>("minimum_post_catchup_duration_sec",
                              config.minimum_post_catchup_duration_sec,
                              config.minimum_post_catchup_duration_sec);
    private_nh_.param<int>("minimum_post_catchup_samples",
                           minimum_post_catchup_samples,
                           minimum_post_catchup_samples);
    config.minimum_post_catchup_samples = static_cast<std::size_t>(
        std::max(1, minimum_post_catchup_samples));
    private_nh_.param<double>("maximum_position_variance_ratio",
                              config.maximum_position_variance_ratio,
                              config.maximum_position_variance_ratio);
    private_nh_.param<double>("maximum_rotation_variance_ratio",
                              config.maximum_rotation_variance_ratio,
                              config.maximum_rotation_variance_ratio);
    private_nh_.param<int>("minimum_consecutive_uncertainty_exceedance_samples",
                           minimum_consecutive_uncertainty_exceedance_samples,
                           minimum_consecutive_uncertainty_exceedance_samples);
    config.minimum_consecutive_uncertainty_exceedance_samples =
        static_cast<std::size_t>(std::max(
            1, minimum_consecutive_uncertainty_exceedance_samples));
    private_nh_.param<bool>("align_candidate_at_boundary",
                            config.align_candidate_at_boundary,
                            config.align_candidate_at_boundary);
    private_nh_.param<bool>("correction_aware_position_bridge_enabled",
                            config.correction_aware_position_bridge_enabled,
                            config.correction_aware_position_bridge_enabled);
    private_nh_.param<double>("position_bridge_duration_sec",
                              config.position_bridge_duration_sec,
                              config.position_bridge_duration_sec);
    private_nh_.param<double>("maximum_added_bridge_speed_mps",
                              config.maximum_added_bridge_speed_mps,
                              config.maximum_added_bridge_speed_mps);
    private_nh_.param<double>("maximum_handover_residual_position_m",
                              config.maximum_handover_residual_position_m,
                              config.maximum_handover_residual_position_m);
    private_nh_.param<bool>("rearm_after_internal_rejection",
                            rearm_after_internal_rejection_, false);
    return config;
  }

  static sm::Pose toPose(const nav_msgs::Odometry &message)
  {
    sm::Pose pose;
    pose.time = message.header.stamp.toSec();
    pose.x = message.pose.pose.position.x;
    pose.y = message.pose.pose.position.y;
    pose.z = message.pose.pose.position.z;
    pose.qx = message.pose.pose.orientation.x;
    pose.qy = message.pose.pose.orientation.y;
    pose.qz = message.pose.pose.orientation.z;
    pose.qw = message.pose.pose.orientation.w;
    pose.frame_id = message.header.frame_id;
    pose.child_frame_id = message.child_frame_id;
    pose.position_variance = message.pose.covariance[0] +
        message.pose.covariance[7] + message.pose.covariance[14];
    pose.rotation_variance = message.pose.covariance[21] +
        message.pose.covariance[28] + message.pose.covariance[35];
    return pose;
  }

  nav_msgs::Odometry::ConstPtr findIncumbent(const double time) const
  {
    if (incumbent_odom_cache_.empty()) return nav_msgs::Odometry::ConstPtr();
    auto upper = incumbent_odom_cache_.lower_bound(time);
    auto best = upper;
    if (upper == incumbent_odom_cache_.end())
      best = std::prev(upper);
    else if (upper != incumbent_odom_cache_.begin())
    {
      const auto lower = std::prev(upper);
      if (std::abs(lower->first - time) <= std::abs(upper->first - time)) best = lower;
    }
    return best->second;
  }

  static double statusDouble(const std::string &status, const std::string &key)
  {
    const std::string token = ";" + key + "=";
    const std::size_t start = status.find(token);
    if (start == std::string::npos)
      return std::numeric_limits<double>::quiet_NaN();
    const std::size_t value_start = start + token.size();
    const std::size_t end = status.find(';', value_start);
    try
    {
      return std::stod(status.substr(value_start, end - value_start));
    }
    catch (const std::exception &)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  static bool allBridgeApprovalsPresent(const std::string &status)
  {
    const char *keys[] = {"healthy_tail_approved", "recovery_benefit_approved",
                          "map_coverage_pass", "map_support_pass"};
    for (const char *key : keys)
    {
      const std::string prefix = std::string(";") + key + "=";
      const std::size_t start = status.find(prefix);
      if (start == std::string::npos || status.find(prefix, start + 1) !=
                                         std::string::npos)
        return false;
      const std::size_t value_start = start + prefix.size();
      const std::size_t end = status.find(';', value_start);
      if (status.substr(value_start, end - value_start) != "1") return false;
    }
    return true;
  }

  static std::string csvText(const std::string &value)
  {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char character : value)
    {
      if (character == '"') escaped.push_back('"');
      escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
  }

  void traceCallback(const std::string &event, const double sensor_time,
                     const nav_msgs::Odometry::ConstPtr &odometry =
                         nav_msgs::Odometry::ConstPtr(),
                     const std::string &status = std::string())
  {
    if (!callback_trace_) return;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto &pose = odometry ? odometry->pose.pose :
        nav_msgs::Odometry().pose.pose;
    std::ostringstream covariance;
    if (odometry)
    {
      covariance << std::setprecision(17);
      for (std::size_t index = 0; index < odometry->pose.covariance.size(); ++index)
      {
        if (index) covariance << ';';
        covariance << odometry->pose.covariance[index];
      }
    }
    callback_trace_ << std::setprecision(17) << ++callback_trace_index_ << ','
        << ros::WallTime::now().toSec() << ',' << event << ',' << sensor_time
        << ',' << sm::stateName(selector_.evaluation().state) << ','
        << latest_published_incumbent_time_ << ',' << last_active_odom_time_
        << ',' << published_odom_watermark_ << ','
        << csvText(odometry ? odometry->header.frame_id : std::string()) << ','
        << csvText(odometry ? odometry->child_frame_id : std::string()) << ','
        << (odometry ? pose.position.x : nan) << ','
        << (odometry ? pose.position.y : nan) << ','
        << (odometry ? pose.position.z : nan) << ','
        << (odometry ? pose.orientation.x : nan) << ','
        << (odometry ? pose.orientation.y : nan) << ','
        << (odometry ? pose.orientation.z : nan) << ','
        << (odometry ? pose.orientation.w : nan) << ','
        << csvText(covariance.str()) << ',' << csvText(status) << '\n';
    callback_trace_.flush();
  }

  static double positionDivergence(const nav_msgs::Odometry &incumbent,
                                   const nav_msgs::Odometry &candidate)
  {
    const double dx = candidate.pose.pose.position.x -
        incumbent.pose.pose.position.x;
    const double dy = candidate.pose.pose.position.y -
        incumbent.pose.pose.position.y;
    const double dz = candidate.pose.pose.position.z -
        incumbent.pose.pose.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  static double rotationDivergenceDeg(const nav_msgs::Odometry &incumbent,
                                      const nav_msgs::Odometry &candidate)
  {
    const auto &a = incumbent.pose.pose.orientation;
    const auto &b = candidate.pose.pose.orientation;
    const double an = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    const double bn = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w);
    if (an <= 1e-12 || bn <= 1e-12) return 180.0;
    double dot = (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w) / (an * bn);
    dot = std::max(-1.0, std::min(1.0, std::abs(dot)));
    return 2.0 * std::acos(dot) * 180.0 / 3.14159265358979323846;
  }

  void logBoundaryDiagnostic(
      const std::string &event, const std::string &status,
      const nav_msgs::Odometry::ConstPtr &candidate,
      const nav_msgs::Odometry::ConstPtr &incumbent)
  {
    if (!boundary_diagnostics_) return;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double candidate_time = candidate ? candidate->header.stamp.toSec() : nan;
    const double incumbent_time = incumbent ? incumbent->header.stamp.toSec() : nan;
    const double signed_dt = candidate && incumbent ?
        candidate_time - incumbent_time : nan;
    const double absolute_dt = std::isfinite(signed_dt) ? std::abs(signed_dt) : nan;
    const double candidate_minus_status = candidate &&
        std::isfinite(commit_status_sensor_time_sec_) ?
        candidate_time - commit_status_sensor_time_sec_ : nan;
    const double candidate_lag = candidate &&
        std::isfinite(latest_published_incumbent_time_) ?
        latest_published_incumbent_time_ - candidate_time : nan;
    const int within_tolerance = std::isfinite(absolute_dt) ?
        static_cast<int>(absolute_dt <= timestampTolerance()) : -1;
    const double position = candidate && incumbent ?
        positionDivergence(*incumbent, *candidate) : nan;
    const double rotation = candidate && incumbent ?
        rotationDivergenceDeg(*incumbent, *candidate) : nan;
    boundary_diagnostics_ << std::setprecision(17)
        << event << ',' << ros::WallTime::now().toSec() << ','
        << csvText(status) << ',' << commit_status_sensor_time_sec_ << ','
        << commit_status_relative_time_sec_ << ',' << candidate_callback_sequence_
        << ',' << latest_candidate_callback_time_ << ','
        << maximum_candidate_time_ << ',' << latest_published_incumbent_time_
        << ',' << candidate_time << ',' << incumbent_time << ',' << signed_dt
        << ',' << absolute_dt << ',' << candidate_minus_status << ','
        << candidate_lag << ',' << within_tolerance << ',' << position << ','
        << rotation << ','
        << (incumbent ? incumbent->pose.pose.position.x : nan) << ','
        << (incumbent ? incumbent->pose.pose.position.y : nan) << ','
        << (incumbent ? incumbent->pose.pose.position.z : nan) << ','
        << (candidate ? candidate->pose.pose.position.x : nan) << ','
        << (candidate ? candidate->pose.pose.position.y : nan) << ','
        << (candidate ? candidate->pose.pose.position.z : nan) << '\n';
    boundary_diagnostics_.flush();
  }

  void incumbentOdom(const nav_msgs::Odometry::ConstPtr &message)
  {
    traceCallback("incumbent_odom", message->header.stamp.toSec(), message);
    if (source_revision_barrier_pending_ &&
        message->header.stamp.toNSec() >
            held_candidate_odom_->header.stamp.toNSec())
      rejectSourceRevisionBarrier("source_revision_incumbent_advanced");
    const double stamp = message->header.stamp.toSec();
    incumbent_odom_cache_[stamp] = message;
    while (!incumbent_odom_cache_.empty() &&
           incumbent_odom_cache_.begin()->first < stamp - cache_horizon_sec_)
      incumbent_odom_cache_.erase(incumbent_odom_cache_.begin());

    if (pendingPublicBridge())
    {
      // The mappers normally publish the same sensor stamp, incumbent first.
      // Reserve one incumbent sample so the candidate can own the next stamp.
      if (pending_incumbent_odom_ &&
          publishActiveOdometry(pending_incumbent_odom_, false))
      {
        last_active_odom_time_ = pending_incumbent_odom_->header.stamp.toSec();
        latest_published_incumbent_time_ = std::max(
            latest_published_incumbent_time_, last_active_odom_time_);
      }
      pending_incumbent_odom_ = message;
      return;
    }

    if (!candidateActive() &&
        publishActiveOdometry(message, false))
    {
      last_active_odom_time_ = stamp;
      latest_published_incumbent_time_ =
          std::max(latest_published_incumbent_time_, stamp);
    }
  }

  void candidateOdom(const nav_msgs::Odometry::ConstPtr &message)
  {
    processCandidateOdom(message, false);
  }

  void processCandidateOdom(const nav_msgs::Odometry::ConstPtr &message,
                            const bool revision_ready)
  {
    if (!revision_ready)
    {
      traceCallback("candidate_odom", message->header.stamp.toSec(), message);
      ++candidate_callback_sequence_;
      latest_candidate_callback_time_ = message->header.stamp.toSec();
      maximum_candidate_time_ = std::max(maximum_candidate_time_,
                                         latest_candidate_callback_time_);
    }
    if (source_revision_barrier_pending_ && !revision_ready)
    {
      if (message->header.stamp.toNSec() >
          held_candidate_odom_->header.stamp.toNSec())
        rejectSourceRevisionBarrier("source_revision_candidate_advanced");
      return;
    }
    const sm::State before = selector_.evaluation().state;
    if (before == sm::State::PROBATION)
    {
      const nav_msgs::Odometry::ConstPtr incumbent =
          findIncumbent(message->header.stamp.toSec());
      if (incumbent)
      {
        const bool precommit = std::isfinite(commit_status_sensor_time_sec_) &&
            message->header.stamp.toSec() + 1e-9 <
                commit_status_sensor_time_sec_;
        logBoundaryDiagnostic(precommit ? "precommit_pose_ignored" :
                                           "probation_pose",
                              std::string(), message, incumbent);
        selector_.observe(toPose(*incumbent), toPose(*message),
                          latest_published_incumbent_time_);
      }
    }
    handleStateChange(before);
    if (selector_.evaluation().state == sm::State::CANDIDATE &&
        selector_.positionBridgeEnabled() && !public_candidate_active_)
    {
      const double stamp = message->header.stamp.toSec();
      const nav_msgs::Odometry::ConstPtr incumbent = findIncumbent(stamp);
      if (!incumbent || std::abs(incumbent->header.stamp.toSec() - stamp) >
                            selector_.timestampToleranceSec())
      {
        if (revision_ready)
          rejectSourceRevisionBarrier("source_revision_boundary_pair_lost");
        return; // Keep the incumbent public until an exact boundary pair exists.
      }
      const sm::Pose bridged = selector_.alignedCandidatePose(toPose(*message));
      const double residual = positionDivergence(*incumbent, *message);
      const double corrected_dx = bridged.x - incumbent->pose.pose.position.x;
      const double corrected_dy = bridged.y - incumbent->pose.pose.position.y;
      const double corrected_dz = bridged.z - incumbent->pose.pose.position.z;
      const double corrected_residual = std::sqrt(
          corrected_dx * corrected_dx + corrected_dy * corrected_dy +
          corrected_dz * corrected_dz);
      if (message->header.frame_id != incumbent->header.frame_id ||
          message->child_frame_id != incumbent->child_frame_id ||
          corrected_residual > selector_.maximumHandoverResidualPositionM())
      {
        selector_.abortBeforePublicCommit("public_boundary_position_limit");
        flushAllPendingIncumbent();
        handleStateChange(sm::State::CANDIDATE);
        return;
      }
      if (stamp <= last_active_odom_time_ + 1e-9 ||
          stamp <= published_odom_watermark_ + 1e-9)
      {
        if (revision_ready)
          rejectSourceRevisionBarrier("source_revision_public_watermark_advanced");
        return;
      }
      if (source_revision_barrier_enabled_ && !revision_ready)
      {
        held_candidate_odom_ = message;
        source_revision_barrier_pending_ = true;
        source_revision_barrier_request_wall_ = ros::WallTime::now().toSec();
        std_msgs::UInt64 request;
        request.data = message->header.stamp.toNSec();
        source_revision_cut_request_pub_.publish(request);
        logBoundaryDiagnostic("source_revision_prepublic_requested", "",
                              message, incumbent);
        return;
      }
      flushPendingIncumbentBefore(stamp);
      selector_.startPositionBridgeRelease(stamp);
      if (revision_ready)
      {
        std_msgs::String manifest;
        manifest.data = source_revision_barrier_manifest_;
        traceCallback("source_revision_manifest_published", stamp,
                      nav_msgs::Odometry::ConstPtr(), manifest.data);
        source_revision_manifest_pub_.publish(manifest);
      }
      if (publishActiveOdometry(message, true))
      {
        if (revision_ready)
          traceCallback("source_revision_public_commit", stamp, message,
                        source_revision_barrier_manifest_);
        clearPendingIncumbent();
        last_active_odom_time_ = stamp;
        switch_time_ = stamp;
        public_candidate_active_ = true;
        public_bridge_frame_id_ = message->header.frame_id;
        if (source_revision_cut_request_enabled_ &&
            !source_revision_barrier_enabled_)
        {
          std_msgs::UInt64 request;
          request.data = message->header.stamp.toNSec();
          source_revision_cut_request_pub_.publish(request);
        }
        if (frame_epoch_diagnostics_enabled_)
        {
          native_frame_id_ = public_bridge_frame_id_ + "_candidate_epoch_" +
              std::to_string(++candidate_epoch_counter_);
          cached_candidate_path_.reset();
          const sm::Pose source_pose = toPose(*message);
          const sm::Pose public_pose = selector_.alignedCandidatePose(source_pose);
          std::ostringstream contract_stream;
          contract_stream << std::setprecision(17)
              << "schema=1;presentation_frame=" << public_bridge_frame_id_
              << ";map_frame=" << native_frame_id_
              << ";native_topics=frame_epoch/candidate_native/*"
              << ";relation=p_presentation=p_native+translation_tf"
              << ";initial_translation_xyz_m="
              << public_pose.x - source_pose.x << ','
              << public_pose.y - source_pose.y << ','
              << public_pose.z - source_pose.z
              << ";release_start_sec=" << stamp
              << ";release_duration_sec=" << selector_.positionBridgeDurationSec()
              << ";historical_pose_rule=initial_translation_times_retained_fraction";
          std_msgs::String contract;
          contract.data = contract_stream.str();
          frame_epoch_contract_pub_.publish(contract);
          if (frame_epoch_snapshot_preview_enabled_)
          {
            std_msgs::String manifest;
            std::ostringstream stream;
            stream << "schema=1;epoch=" << candidate_epoch_counter_
                   << ";native_frame=" << native_frame_id_
                   << ";presentation_frame=" << public_bridge_frame_id_
                   << ";native_path_topic=/shadow_output_mux/frame_epoch/candidate_native/path"
                   << ";native_map_topic=/shadow_output_mux/frame_epoch/candidate_native/map"
                   << ";preview_path_topic=/shadow_output_mux/frame_epoch/snapshot_preview/path"
                   << ";preview_map_topic=/shadow_output_mux/frame_epoch/snapshot_preview/map"
                   << ";snapshot_transform_rule=map_message_stamp"
                   << ";canonical_public_unchanged=1"
                   << ";canonical_public_history_replacement_claim=0";
            manifest.data = stream.str();
            frame_epoch_versioned_manifest_pub_.publish(manifest);
          }
          publishEpochOdometry(*message);
        }
        first_public_boundary_raw_position_m_ = residual;
        first_public_boundary_corrected_position_m_ = corrected_residual;
        publishEvaluation();
        ROS_WARN_STREAM("[ ShadowMux ] correction-aware public bridge started at "
                        << stamp << " residual=" << corrected_residual);
      }
      else if (revision_ready)
        rejectSourceRevisionBarrier("source_revision_public_publish_failed");
      return;
    }
    if (selector_.evaluation().state == sm::State::CANDIDATE &&
        message->header.stamp.toSec() > last_active_odom_time_ + 1e-9 &&
        publishActiveOdometry(message, true))
    {
      last_active_odom_time_ = message->header.stamp.toSec();
      publishEpochOdometry(*message);
    }
  }

  static std::map<std::string, std::string> sourceRevisionFields(
      const std::string &value, const char separator)
  {
    std::map<std::string, std::string> fields;
    std::istringstream input(value);
    std::string token;
    while (std::getline(input, token, separator))
    {
      const std::size_t equal = token.find('=');
      if (equal == std::string::npos || equal == 0 || equal + 1 == token.size() ||
          !fields.emplace(token.substr(0, equal),
                          token.substr(equal + 1)).second)
        throw std::runtime_error("malformed or duplicate source revision field");
    }
    return fields;
  }

  static std::map<std::string, std::string> sourceRevisionFile(
      const std::filesystem::path &path)
  {
    std::ifstream input(path);
    if (!input)
      throw std::runtime_error("source revision file unavailable");
    std::ostringstream content;
    content << input.rdbuf();
    return sourceRevisionFields(content.str(), '\n');
  }

  void rejectSourceRevisionBarrier(const std::string &reason)
  {
    if (!source_revision_barrier_pending_ && !held_candidate_odom_ &&
        source_revision_barrier_manifest_.empty())
      return;
    source_revision_barrier_pending_ = false;
    held_candidate_odom_.reset();
    source_revision_barrier_manifest_.clear();
    const sm::State before = selector_.evaluation().state;
    if (before == sm::State::CANDIDATE && !public_candidate_active_)
    {
      selector_.abortBeforePublicCommit(reason);
      flushAllPendingIncumbent();
      handleStateChange(before);
    }
    ROS_ERROR_STREAM("[ ShadowMux ] prepublic source revision rejected: " << reason);
  }

  void sourceRevisionBarrierTimer(const ros::WallTimerEvent &)
  {
    if (source_revision_barrier_pending_ &&
        ros::WallTime::now().toSec() - source_revision_barrier_request_wall_ >
            source_revision_barrier_timeout_sec_)
      rejectSourceRevisionBarrier("source_revision_timeout");
  }

  void sourceRevisionReady(const std_msgs::String::ConstPtr &message)
  {
    traceCallback("source_revision_ready", std::numeric_limits<double>::quiet_NaN(),
                  nav_msgs::Odometry::ConstPtr(), message->data);
    if (!source_revision_barrier_pending_ || !held_candidate_odom_ ||
        selector_.evaluation().state != sm::State::CANDIDATE)
      return;
    const std::uint64_t held_ns = held_candidate_odom_->header.stamp.toNSec();
    try
    {
      if (ros::WallTime::now().toSec() - source_revision_barrier_request_wall_ >
          source_revision_barrier_timeout_sec_)
        throw std::runtime_error("source_revision_timeout");
      const auto fields = sourceRevisionFields(message->data, ';');
      const auto required = [&fields](const std::string &key) -> const std::string & {
        const auto found = fields.find(key);
        if (found == fields.end())
          throw std::runtime_error("missing source revision field: " + key);
        return found->second;
      };
      const std::string expected_schema =
          source_revision_payload_hash_enabled_ ? "2" : "1";
      if (required("schema") != expected_schema ||
          std::stoull(required("requested_public_switch_ns")) != held_ns)
        throw std::runtime_error("source revision request mismatch");
      if (required("status") != "ready")
        throw std::runtime_error("source_revision_mapper_failed");
      const std::filesystem::path receipt_path(required("receipt"));
      const std::filesystem::path marker_path(required("completed"));
      if (receipt_path.filename() != "source_revision.cut" ||
          marker_path.filename() != "cut_revision.completed" ||
          receipt_path.parent_path() != marker_path.parent_path() ||
          receipt_path.parent_path().filename() !=
              "cut_revision_" + std::to_string(held_ns))
        throw std::runtime_error("source revision paths are inconsistent");
      const auto receipt = sourceRevisionFile(receipt_path);
      const auto marker = sourceRevisionFile(marker_path);
      const auto receiptField = [&receipt](const std::string &key) -> const std::string & {
        const auto found = receipt.find(key);
        if (found == receipt.end())
          throw std::runtime_error("missing source receipt field: " + key);
        return found->second;
      };
      const auto marker_revision = marker.find("revision_id");
      if (marker_revision == marker.end() ||
          receiptField("schema") != expected_schema ||
          receiptField("scope") != "mapper_prepublic_cut_for_mux_barrier" ||
          receiptField("source_frame") != "camera_init" ||
          receiptField("revision_id") != required("revision_id") ||
          marker_revision->second != required("revision_id") ||
          std::stoull(receiptField("requested_public_switch_ns")) != held_ns ||
          std::stoull(receiptField("sensor_watermark_ns")) != held_ns ||
          std::stoull(receiptField("path_final_ns")) != held_ns ||
          std::stoull(receiptField("path_pose_count")) == 0 ||
          std::stoull(receiptField("map_point_count")) == 0)
        throw std::runtime_error("source revision content/watermark mismatch");
      const std::filesystem::path map_path(receiptField("raw_map"));
      const std::filesystem::path path_path(receiptField("source_path"));
      if (map_path.parent_path() != receipt_path.parent_path() ||
          path_path.parent_path() != receipt_path.parent_path() ||
          map_path.filename() != "full_map.pcd" ||
          path_path.filename() != "source_path.txt" ||
          !std::filesystem::is_regular_file(map_path) ||
          !std::filesystem::is_regular_file(path_path) ||
          std::filesystem::file_size(map_path) == 0 ||
          std::filesystem::file_size(path_path) == 0)
        throw std::runtime_error("source revision payload unavailable");
      if (source_revision_payload_hash_enabled_)
      {
        const auto verify = [&required](const std::filesystem::path &file,
                                        const std::string &name) {
          fast_livo::source_revision::requireSha256(file, required(name));
        };
        if (receiptField("raw_map_sha256") != required("raw_map_sha256") ||
            receiptField("source_path_sha256") != required("source_path_sha256"))
          throw std::runtime_error("source revision receipt/ready digest mismatch");
        verify(map_path, "raw_map_sha256");
        verify(path_path, "source_path_sha256");
        verify(receipt_path, "receipt_sha256");
        verify(marker_path, "completed_sha256");
        if (ros::WallTime::now().toSec() - source_revision_barrier_request_wall_ >
            source_revision_barrier_timeout_sec_)
          throw std::runtime_error("source_revision_timeout_during_hash_verification");
      }
      source_revision_barrier_manifest_ =
          "schema=" + expected_schema +
          ";status=ready;scope=prepublic_durable_file_reference"
          ";revision_id=" + required("revision_id") +
          ";public_switch_ns=" + std::to_string(held_ns) +
          ";source_watermark_ns=" + receiptField("sensor_watermark_ns") +
          ";source_frame=" + receiptField("source_frame") +
          ";raw_map=" + map_path.string() +
          ";source_path=" + path_path.string() +
          ";receipt=" + receipt_path.string() +
          ";completed=" + marker_path.string() +
          ";atomic_ros_publication=0";
      if (source_revision_payload_hash_enabled_)
        source_revision_barrier_manifest_ +=
            ";raw_map_sha256=" + required("raw_map_sha256") +
            ";source_path_sha256=" + required("source_path_sha256") +
            ";receipt_sha256=" + required("receipt_sha256") +
            ";completed_sha256=" + required("completed_sha256") +
            ";hash_verified_before_public_switch=1";
      const nav_msgs::Odometry::ConstPtr held = held_candidate_odom_;
      source_revision_barrier_pending_ = false;
      held_candidate_odom_.reset();
      logBoundaryDiagnostic("source_revision_prepublic_ready", message->data,
                            held, findIncumbent(held->header.stamp.toSec()));
      processCandidateOdom(held, true);
    }
    catch (const std::exception &error)
    {
      rejectSourceRevisionBarrier(error.what());
    }
  }

  void candidateStatus(const std_msgs::String::ConstPtr &message)
  {
    traceCallback("candidate_status", std::numeric_limits<double>::quiet_NaN(),
                  nav_msgs::Odometry::ConstPtr(), message->data);
    const sm::State before = selector_.evaluation().state;
    bool force_evaluation_publish = false;
    if (message->data.find("candidate_committed") == 0)
    {
      const std::string detail_token = ";detail=";
      const std::size_t detail_start = message->data.find(detail_token);
      if (detail_start != std::string::npos)
        candidate_validation_detail_ = message->data.substr(
            detail_start + detail_token.size());
      commit_status_sensor_time_sec_ = statusDouble(message->data,
                                                     "sensor_time_sec");
      commit_status_relative_time_sec_ = statusDouble(message->data,
                                                       "relative_time_sec");
      logBoundaryDiagnostic("candidate_committed", message->data,
                            nav_msgs::Odometry::ConstPtr(),
                            nav_msgs::Odometry::ConstPtr());
      if (selector_.positionBridgeEnabled() &&
          !allBridgeApprovalsPresent(message->data))
      {
        selector_.candidateRejected("bridge_internal_approval_missing");
        force_evaluation_publish = true;
      }
      else
      {
        selector_.candidateCommitted(commit_status_sensor_time_sec_,
                                     latest_published_incumbent_time_);
      }
    }
    else if (message->data.find("candidate_rejected") == 0 ||
             message->data.find("rolled_back") == 0)
    {
      if (source_revision_barrier_pending_)
      {
        source_revision_barrier_pending_ = false;
        held_candidate_odom_.reset();
        source_revision_barrier_manifest_.clear();
      }
      logBoundaryDiagnostic("candidate_rejected", message->data,
                            nav_msgs::Odometry::ConstPtr(),
                            nav_msgs::Odometry::ConstPtr());
      awaiting_internal_rearm_ = rearm_after_internal_rejection_;
      if (selector_.positionBridgeEnabled() &&
          selector_.evaluation().state == sm::State::CANDIDATE &&
          !public_candidate_active_)
      {
        selector_.abortBeforePublicCommit("internal_rejection_before_public_commit");
        flushAllPendingIncumbent();
      }
      else
        selector_.candidateRejected(message->data);
      force_evaluation_publish = true;
    }
    else if (message->data.find("candidate_rearmed") == 0 &&
             rearm_after_internal_rejection_ &&
             selector_.evaluation().state == sm::State::INCUMBENT &&
             selector_.evaluation().candidate_rejected)
    {
      logBoundaryDiagnostic("candidate_rearmed", message->data,
                            nav_msgs::Odometry::ConstPtr(),
                            nav_msgs::Odometry::ConstPtr());
      selector_.resetForNextCandidate();
      candidate_validation_detail_.clear();
      commit_status_sensor_time_sec_ =
          std::numeric_limits<double>::quiet_NaN();
      commit_status_relative_time_sec_ =
          std::numeric_limits<double>::quiet_NaN();
      awaiting_internal_rearm_ = false;
      force_evaluation_publish = true;
    }
    handleStateChange(before, force_evaluation_publish);
  }

  void finalize(const std_msgs::Bool::ConstPtr &message)
  {
    traceCallback("finalize", std::numeric_limits<double>::quiet_NaN(),
                  nav_msgs::Odometry::ConstPtr(), message->data ? "true" : "false");
    if (!message->data) return;
    if (source_revision_barrier_pending_)
    {
      source_revision_barrier_pending_ = false;
      held_candidate_odom_.reset();
      source_revision_barrier_manifest_.clear();
    }
    const sm::State before = selector_.evaluation().state;
    if (selector_.positionBridgeEnabled() && before == sm::State::CANDIDATE &&
        !public_candidate_active_)
    {
      selector_.abortBeforePublicCommit("public_boundary_pair_unavailable_at_end");
      flushAllPendingIncumbent();
    }
    else
      selector_.finalize();
    handleStateChange(before);
  }

  void handleStateChange(const sm::State before, const bool force_publish = false)
  {
    const sm::State after = selector_.evaluation().state;
    if (before == after && !force_publish) return;
    publishEvaluation();
    if (before == after) return;
    if (after == sm::State::ABORTED)
    {
      if (!rearm_after_internal_rejection_ || !awaiting_internal_rearm_)
      {
        std_msgs::Bool abort;
        abort.data = true;
        abort_pub_.publish(abort);
      }
      ROS_ERROR_STREAM("[ ShadowMux ] candidate aborted: "
                       << selector_.evaluation().reason);
    }
    else if (after == sm::State::CANDIDATE)
    {
      if (selector_.positionBridgeEnabled())
        ROS_WARN_STREAM("[ ShadowMux ] candidate approved; waiting for public boundary pair");
      else
      {
        last_active_odom_time_ = latest_published_incumbent_time_;
        switch_time_ = latest_published_incumbent_time_;
        ROS_WARN_STREAM("[ ShadowMux ] atomically selected candidate generation");
      }
    }
    else if (after == sm::State::PROBATION)
    {
      ROS_WARN_STREAM("[ ShadowMux ] candidate committed internally; probation started");
    }
    else if (after == sm::State::INCUMBENT)
    {
      if (selector_.evaluation().candidate_rejected)
        ROS_WARN_STREAM("[ ShadowMux ] candidate rejected; incumbent remains active: "
                        << selector_.evaluation().candidate_rejection_reason);
      else
        ROS_WARN_STREAM("[ ShadowMux ] candidate re-armed; incumbent remains active");
    }
  }

  void publishEvaluation()
  {
    const sm::Evaluation &value = selector_.evaluation();
    std::ostringstream stream;
    stream << std::setprecision(17)
           << "state=" << (selector_.positionBridgeEnabled() &&
                             value.state == sm::State::CANDIDATE &&
                             !public_candidate_active_ ? "probation" :
                             sm::stateName(value.state))
           << ";reason=" << value.reason
           << ";candidate_rejected="
           << static_cast<int>(value.candidate_rejected)
           << ";candidate_rejection_reason="
           << value.candidate_rejection_reason
           << ";matched=" << value.matched_samples
           << ";boundary_position_m=" << value.boundary_position_m
           << ";boundary_rotation_deg=" << value.boundary_rotation_deg
           << ";raw_boundary_position_m=" << value.raw_boundary_position_m
           << ";raw_boundary_rotation_deg=" << value.raw_boundary_rotation_deg
           << ";boundary_alignment_applied="
           << static_cast<int>(value.boundary_alignment_applied)
           << ";maximum_position_growth_m=" << value.maximum_position_growth_m
           << ";maximum_rotation_growth_deg=" << value.maximum_rotation_growth_deg
           << ";probation_elapsed_sec=" << value.probation_elapsed_sec
           << ";post_catchup_samples=" << value.post_catchup_samples
           << ";post_catchup_elapsed_sec=" << value.post_catchup_elapsed_sec
           << ";maximum_position_variance_ratio="
           << value.maximum_position_variance_ratio
           << ";maximum_rotation_variance_ratio="
           << value.maximum_rotation_variance_ratio
           << ";second_maximum_position_variance_ratio="
           << value.second_maximum_position_variance_ratio
           << ";second_maximum_rotation_variance_ratio="
           << value.second_maximum_rotation_variance_ratio
           << ";mean_position_variance_ratio="
           << value.mean_position_variance_ratio
           << ";mean_rotation_variance_ratio="
           << value.mean_rotation_variance_ratio
           << ";uncertainty_samples=" << value.uncertainty_samples
           << ";required_consecutive_uncertainty_exceedance_samples="
           << value.required_consecutive_uncertainty_exceedance_samples
           << ";position_uncertainty_exceedance_samples="
           << value.position_uncertainty_exceedance_samples
           << ";rotation_uncertainty_exceedance_samples="
           << value.rotation_uncertainty_exceedance_samples
           << ";current_consecutive_position_uncertainty_exceedances="
           << value.current_consecutive_position_uncertainty_exceedances
           << ";current_consecutive_rotation_uncertainty_exceedances="
           << value.current_consecutive_rotation_uncertainty_exceedances
           << ";maximum_consecutive_position_uncertainty_exceedances="
           << value.maximum_consecutive_position_uncertainty_exceedances
           << ";maximum_consecutive_rotation_uncertainty_exceedances="
           << value.maximum_consecutive_rotation_uncertainty_exceedances
           << ";position_variance_ratio_time_sec="
           << value.position_variance_ratio_time_sec
           << ";candidate_position_variance_at_maximum_ratio="
           << value.candidate_position_variance_at_maximum_ratio
           << ";incumbent_position_variance_at_maximum_ratio="
           << value.incumbent_position_variance_at_maximum_ratio
           << ";rotation_variance_ratio_time_sec="
           << value.rotation_variance_ratio_time_sec
           << ";candidate_rotation_variance_at_maximum_ratio="
           << value.candidate_rotation_variance_at_maximum_ratio
           << ";incumbent_rotation_variance_at_maximum_ratio="
           << value.incumbent_rotation_variance_at_maximum_ratio
           << ";candidate_commit_sensor_time_sec="
           << value.candidate_commit_sensor_time_sec
           << ";candidate_commit_latest_incumbent_time_sec="
           << value.candidate_commit_latest_incumbent_time_sec
           << ";candidate_commit_lag_sec=" << value.candidate_commit_lag_sec
           << ";precommit_candidate_samples_ignored="
           << value.precommit_candidate_samples_ignored
           << ";last_ignored_precommit_candidate_time_sec="
           << value.last_ignored_precommit_candidate_time_sec
           << ";first_evaluated_candidate_time_sec="
           << value.first_evaluated_candidate_time_sec
           << ";first_evaluated_incumbent_time_sec="
           << value.first_evaluated_incumbent_time_sec
           << ";first_evaluated_match_dt_sec="
           << value.first_evaluated_match_dt_sec
           << ";first_evaluated_candidate_minus_commit_sec="
           << value.first_evaluated_candidate_minus_commit_sec
           << ";maximum_absolute_match_dt_sec="
           << value.maximum_absolute_match_dt_sec;
    if (selector_.positionBridgeEnabled())
      stream << ";selector_state=" << sm::stateName(value.state)
             << ";public_candidate_active=" << static_cast<int>(candidateActive())
             << ";position_bridge_applied="
             << static_cast<int>(value.position_bridge_applied)
             << ";position_bridge_initial_offset_m="
             << value.position_bridge_initial_offset_m
             << ";position_bridge_peak_added_speed_mps="
             << value.position_bridge_peak_added_speed_mps
             << ";position_bridge_release_start_sec="
             << value.position_bridge_release_start_sec
             << ";handover_residual_position_m="
             << value.handover_residual_position_m
             << ";first_public_boundary_raw_position_m="
             << first_public_boundary_raw_position_m_
             << ";first_public_boundary_corrected_position_m="
             << first_public_boundary_corrected_position_m_;
    if (!candidate_validation_detail_.empty())
      stream << ';' << candidate_validation_detail_;
    std_msgs::String message;
    message.data = stream.str();
    generation_pub_.publish(message);
  }

  double timestampTolerance() const { return selector_.timestampToleranceSec(); }

  bool publishActiveOdometry(const nav_msgs::Odometry::ConstPtr &message,
                             const bool align_candidate)
  {
    const double stamp = message->header.stamp.toSec();
    if (!advanceWatermark(stamp, published_odom_watermark_))
    {
      ++dropped_nonmonotonic_odom_;
      ROS_WARN_STREAM_THROTTLE(2.0,
          "[ ShadowMux ] dropped non-monotonic odometry stamp=" << stamp
          << " watermark=" << published_odom_watermark_
          << " count=" << dropped_nonmonotonic_odom_);
      return false;
    }
    // Freeze the small output message before returning from the subscriber
    // callback. Forwarding the incoming shared pointer can defer serialization
    // and obscures the publication-order contract under heavy dual-mapper load.
    nav_msgs::Odometry output = *message;
    if (align_candidate && selector_.boundaryAlignmentInitialized())
    {
      const sm::Pose aligned = selector_.alignedCandidatePose(toPose(output));
      output.pose.pose.position.x = aligned.x;
      output.pose.pose.position.y = aligned.y;
      output.pose.pose.position.z = aligned.z;
      output.pose.pose.orientation.x = aligned.qx;
      output.pose.pose.orientation.y = aligned.qy;
      output.pose.pose.orientation.z = aligned.qz;
      output.pose.pose.orientation.w = aligned.qw;
    }
    active_odom_pub_.publish(output);
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(output.pose.pose.position.x,
                                    output.pose.pose.position.y,
                                    output.pose.pose.position.z));
    transform.setRotation(tf::Quaternion(output.pose.pose.orientation.x,
                                         output.pose.pose.orientation.y,
                                         output.pose.pose.orientation.z,
                                         output.pose.pose.orientation.w));
    transform_broadcaster_.sendTransform(tf::StampedTransform(
        transform, output.header.stamp, output.header.frame_id,
        output.child_frame_id));
    return true;
  }

  void publishEpochOdometry(const nav_msgs::Odometry &message)
  {
    if (!frame_epoch_diagnostics_enabled_ || !public_candidate_active_ ||
        native_frame_id_.empty()) return;
    try
    {
      nav_msgs::Odometry presentation = message;
      const sm::Pose aligned = selector_.alignedCandidatePose(toPose(message));
      presentation.pose.pose.position.x = aligned.x;
      presentation.pose.pose.position.y = aligned.y;
      presentation.pose.pose.position.z = aligned.z;
      const auto translation = fast_livo::frame_epoch::presentationFromNative(
          message, presentation);
      const auto native = fast_livo::frame_epoch::nativeOdometry(
          message, public_bridge_frame_id_, native_frame_id_);
      native_odom_pub_.publish(native);
      tf::Transform transform;
      transform.setOrigin(tf::Vector3(translation.x, translation.y, translation.z));
      transform.setRotation(tf::Quaternion(0.0, 0.0, 0.0, 1.0));
      transform_broadcaster_.sendTransform(tf::StampedTransform(
          transform, message.header.stamp, public_bridge_frame_id_,
          native_frame_id_));
    }
    catch (const std::exception &error)
    {
      ROS_ERROR_STREAM_THROTTLE(2.0,
          "[ ShadowMux ] suppressing frame-epoch diagnostic odometry: " << error.what());
    }
  }

  void publishEpochPath(const nav_msgs::Path &message)
  {
    if (!frame_epoch_diagnostics_enabled_ || native_frame_id_.empty()) return;
    try
    {
      native_path_pub_.publish(fast_livo::frame_epoch::nativePath(
          message, public_bridge_frame_id_, native_frame_id_));
    }
    catch (const std::exception &error)
    {
      ROS_ERROR_STREAM_THROTTLE(2.0,
          "[ ShadowMux ] suppressing frame-epoch diagnostic path: " << error.what());
    }
  }

  void publishEpochCloud(const sensor_msgs::PointCloud2 &message,
                         const ros::Publisher &publisher)
  {
    if (!frame_epoch_diagnostics_enabled_ || native_frame_id_.empty()) return;
    try
    {
      publisher.publish(fast_livo::frame_epoch::nativeCloud(
          message, public_bridge_frame_id_, native_frame_id_));
    }
    catch (const std::exception &error)
    {
      ROS_ERROR_STREAM_THROTTLE(2.0,
          "[ ShadowMux ] suppressing frame-epoch diagnostic cloud/map: " << error.what());
    }
  }

  void publishSnapshotPreview(const sensor_msgs::PointCloud2 &source_map,
                              const sensor_msgs::PointCloud2 &presentation_map)
  {
    if (!frame_epoch_snapshot_preview_enabled_ || !cached_candidate_path_ ||
        native_frame_id_.empty()) return;
    const double map_time = source_map.header.stamp.toSec();
    const double path_time = cached_candidate_path_->header.stamp.toSec();
    if (!std::isfinite(map_time) || !std::isfinite(path_time) ||
        path_time > map_time + 1e-9 || map_time - path_time > 0.2)
    {
      ROS_WARN_STREAM_THROTTLE(2.0,
          "[ ShadowMux ] suppressing snapshot preview: no path within 0.2 s before map");
      return;
    }
    try
    {
      sm::Pose origin;
      origin.time = map_time;
      const sm::Pose aligned = selector_.alignedCandidatePose(origin);
      fast_livo::frame_epoch::Translation translation;
      translation.x = aligned.x - origin.x;
      translation.y = aligned.y - origin.y;
      translation.z = aligned.z - origin.z;
      const nav_msgs::Path preview_path = fast_livo::frame_epoch::snapshotPath(
          *cached_candidate_path_, public_bridge_frame_id_,
          source_map.header.stamp, translation);
      if (presentation_map.header.frame_id != public_bridge_frame_id_ ||
          presentation_map.header.stamp != preview_path.header.stamp ||
          presentation_map.data.empty())
        throw std::invalid_argument("snapshot preview map does not match path");
      snapshot_preview_path_pub_.publish(preview_path);
      snapshot_preview_map_pub_.publish(presentation_map);
      std_msgs::String bundle;
      std::ostringstream stream;
      stream << std::setprecision(17)
             << "schema=1;epoch=" << candidate_epoch_counter_
             << ";native_frame=" << native_frame_id_
             << ";snapshot_time_sec=" << map_time
             << ";source_path_time_sec=" << path_time
             << ";translation_xyz_m=" << translation.x << ','
             << translation.y << ',' << translation.z
             << ";path_pose_count=" << preview_path.poses.size()
             << ";map_data_bytes=" << presentation_map.data.size();
      bundle.data = stream.str();
      snapshot_preview_bundle_pub_.publish(bundle);
    }
    catch (const std::exception &error)
    {
      ROS_ERROR_STREAM_THROTTLE(2.0,
          "[ ShadowMux ] suppressing snapshot preview: " << error.what());
    }
  }

  bool candidateActive() const
  {
    return selector_.evaluation().state == sm::State::CANDIDATE &&
        (!selector_.positionBridgeEnabled() || public_candidate_active_);
  }

  bool pendingPublicBridge() const
  {
    return selector_.positionBridgeEnabled() &&
        selector_.evaluation().state == sm::State::CANDIDATE &&
        !public_candidate_active_;
  }

  void clearPendingIncumbent()
  {
    pending_incumbent_odom_.reset();
    pending_incumbent_path_.reset();
    pending_incumbent_cloud_.reset();
    pending_incumbent_map_.reset();
  }

  void flushPendingIncumbentBefore(const double stamp)
  {
    if (pending_incumbent_path_ &&
        pending_incumbent_path_->header.stamp.toSec() < stamp - 1e-9 &&
        advanceWatermark(pending_incumbent_path_->header.stamp.toSec(),
                         last_active_path_time_))
      active_path_pub_.publish(pending_incumbent_path_);
    if (pending_incumbent_cloud_ &&
        pending_incumbent_cloud_->header.stamp.toSec() < stamp - 1e-9 &&
        advanceWatermark(pending_incumbent_cloud_->header.stamp.toSec(),
                         last_active_cloud_time_))
      active_cloud_pub_.publish(pending_incumbent_cloud_);
    if (pending_incumbent_map_ &&
        pending_incumbent_map_->header.stamp.toSec() < stamp - 1e-9 &&
        advanceWatermark(pending_incumbent_map_->header.stamp.toSec(),
                         last_active_map_time_))
      active_map_pub_.publish(pending_incumbent_map_);
  }

  void flushAllPendingIncumbent()
  {
    if (pending_incumbent_odom_ &&
        publishActiveOdometry(pending_incumbent_odom_, false))
    {
      last_active_odom_time_ = pending_incumbent_odom_->header.stamp.toSec();
      latest_published_incumbent_time_ = std::max(
          latest_published_incumbent_time_, last_active_odom_time_);
    }
    flushPendingIncumbentBefore(std::numeric_limits<double>::infinity());
    clearPendingIncumbent();
  }

  static bool advanceWatermark(const double stamp, double &watermark)
  {
    if (!std::isfinite(stamp) || stamp <= watermark + 1e-9) return false;
    watermark = stamp;
    return true;
  }

  bool candidateModalityAfterSwitch(const double stamp) const
  {
    return stamp > switch_time_ + 1e-9 ||
        (selector_.positionBridgeEnabled() &&
         stamp + 1e-9 >= switch_time_);
  }

  void incumbentPath(const nav_msgs::Path::ConstPtr &message)
  {
    traceCallback("incumbent_path", message->header.stamp.toSec());
    if (pendingPublicBridge())
    {
      if (pending_incumbent_path_ &&
          advanceWatermark(pending_incumbent_path_->header.stamp.toSec(),
                           last_active_path_time_))
        active_path_pub_.publish(pending_incumbent_path_);
      pending_incumbent_path_ = message;
      return;
    }
    if (!candidateActive() &&
        advanceWatermark(message->header.stamp.toSec(), last_active_path_time_))
      active_path_pub_.publish(message);
  }
  void candidatePath(const nav_msgs::Path::ConstPtr &message)
  {
    traceCallback("candidate_path", message->header.stamp.toSec());
    if (candidateActive() && selector_.positionBridgeEnabled())
    {
      if (message->header.frame_id != public_bridge_frame_id_ ||
          std::any_of(message->poses.begin(), message->poses.end(),
                      [this](const geometry_msgs::PoseStamped &pose) {
                        return !pose.header.frame_id.empty() &&
                            pose.header.frame_id != public_bridge_frame_id_;
                      }))
      {
        ROS_ERROR_STREAM_THROTTLE(2.0,
            "[ ShadowMux ] suppressing candidate path with inconsistent frame");
        return;
      }
    }
    if (candidateActive() &&
        candidateModalityAfterSwitch(message->header.stamp.toSec()) &&
        advanceWatermark(message->header.stamp.toSec(), last_active_path_time_))
    {
      nav_msgs::Path output = *message;
      if (selector_.boundaryAlignmentInitialized())
      {
        for (auto &stamped : output.poses)
        {
          sm::Pose pose;
          pose.time = stamped.header.stamp.isZero() ?
              message->header.stamp.toSec() : stamped.header.stamp.toSec();
          pose.x = stamped.pose.position.x;
          pose.y = stamped.pose.position.y;
          pose.z = stamped.pose.position.z;
          pose.qx = stamped.pose.orientation.x;
          pose.qy = stamped.pose.orientation.y;
          pose.qz = stamped.pose.orientation.z;
          pose.qw = stamped.pose.orientation.w;
          const sm::Pose aligned = selector_.alignedCandidatePose(pose);
          stamped.pose.position.x = aligned.x;
          stamped.pose.position.y = aligned.y;
          stamped.pose.position.z = aligned.z;
          stamped.pose.orientation.x = aligned.qx;
          stamped.pose.orientation.y = aligned.qy;
          stamped.pose.orientation.z = aligned.qz;
          stamped.pose.orientation.w = aligned.qw;
        }
      }
      active_path_pub_.publish(output);
      publishEpochPath(*message);
      if (frame_epoch_snapshot_preview_enabled_)
        cached_candidate_path_ = message;
    }
  }
  void incumbentCloud(const sensor_msgs::PointCloud2::ConstPtr &message)
  {
    traceCallback("incumbent_cloud", message->header.stamp.toSec());
    if (selector_.positionBridgeEnabled() && message->data.empty())
      return; // FAST-LIVO2 sends an empty heartbeat before the real cloud.
    if (pendingPublicBridge())
    {
      if (pending_incumbent_cloud_ &&
          advanceWatermark(pending_incumbent_cloud_->header.stamp.toSec(),
                           last_active_cloud_time_))
        active_cloud_pub_.publish(pending_incumbent_cloud_);
      pending_incumbent_cloud_ = message;
      return;
    }
    if (!candidateActive() &&
        advanceWatermark(message->header.stamp.toSec(), last_active_cloud_time_))
      active_cloud_pub_.publish(message);
  }
  void candidateCloud(const sensor_msgs::PointCloud2::ConstPtr &message)
  {
    traceCallback("candidate_cloud", message->header.stamp.toSec());
    if (selector_.positionBridgeEnabled() && message->data.empty())
      return;
    if (candidateActive() && selector_.positionBridgeEnabled() &&
        message->header.frame_id != public_bridge_frame_id_)
    {
      ROS_ERROR_STREAM_THROTTLE(2.0,
          "[ ShadowMux ] suppressing candidate cloud with inconsistent frame");
      return;
    }
    if (candidateActive() &&
        candidateModalityAfterSwitch(message->header.stamp.toSec()) &&
        (selector_.positionBridgeEnabled() ||
         advanceWatermark(message->header.stamp.toSec(), last_active_cloud_time_)))
    {
      const sensor_msgs::PointCloud2 output = alignedCandidateCloud(*message);
      if (selector_.positionBridgeEnabled())
      {
        if (output.data.empty() ||
            !advanceWatermark(message->header.stamp.toSec(),
                              last_active_cloud_time_))
          return;
      }
      active_cloud_pub_.publish(output);
      publishEpochCloud(*message, native_cloud_pub_);
    }
  }
  void incumbentMap(const sensor_msgs::PointCloud2::ConstPtr &message)
  {
    traceCallback("incumbent_map", message->header.stamp.toSec());
    if (selector_.positionBridgeEnabled() && message->data.empty())
      return;
    if (pendingPublicBridge())
    {
      if (pending_incumbent_map_ &&
          advanceWatermark(pending_incumbent_map_->header.stamp.toSec(),
                           last_active_map_time_))
        active_map_pub_.publish(pending_incumbent_map_);
      pending_incumbent_map_ = message;
      return;
    }
    if (!candidateActive() &&
        advanceWatermark(message->header.stamp.toSec(), last_active_map_time_))
      active_map_pub_.publish(message);
  }
  void candidateMap(const sensor_msgs::PointCloud2::ConstPtr &message)
  {
    traceCallback("candidate_map", message->header.stamp.toSec());
    if (selector_.positionBridgeEnabled() && message->data.empty())
      return;
    if (candidateActive() && selector_.positionBridgeEnabled() &&
        message->header.frame_id != public_bridge_frame_id_)
    {
      ROS_ERROR_STREAM_THROTTLE(2.0,
          "[ ShadowMux ] suppressing candidate map with inconsistent frame");
      return;
    }
    if (candidateActive() &&
        candidateModalityAfterSwitch(message->header.stamp.toSec()) &&
        (selector_.positionBridgeEnabled() ||
         advanceWatermark(message->header.stamp.toSec(), last_active_map_time_)))
    {
      const sensor_msgs::PointCloud2 output = alignedCandidateCloud(*message);
      if (selector_.positionBridgeEnabled())
      {
        if (output.data.empty() ||
            !advanceWatermark(message->header.stamp.toSec(),
                              last_active_map_time_))
          return;
      }
      active_map_pub_.publish(output);
      publishEpochCloud(*message, native_map_pub_);
      publishSnapshotPreview(*message, output);
    }
  }

  sensor_msgs::PointCloud2 alignedCandidateCloud(
      const sensor_msgs::PointCloud2 &message) const
  {
    sensor_msgs::PointCloud2 output = message;
    if (!selector_.boundaryAlignmentInitialized() || output.data.empty())
      return output;

    // Some FAST-LIVO2 publications use an intentionally empty PointCloud2 as
    // a heartbeat.  PointCloud2Iterator throws when such a message has no
    // schema; there is no geometry to transform in that case.  A non-empty
    // cloud without Cartesian fields is unsafe to publish after a frame
    // change, so fail closed by suppressing its payload.
    const auto has_field = [&output](const std::string &name) {
      return std::any_of(output.fields.begin(), output.fields.end(),
                         [&name](const sensor_msgs::PointField &field) {
                           return field.name == name;
                         });
    };
    if (!has_field("x") || !has_field("y") || !has_field("z"))
    {
      ROS_ERROR_STREAM_THROTTLE(
          2.0, "[ ShadowMux ] suppressing non-empty candidate cloud without x/y/z fields");
      output.data.clear();
      output.width = 0;
      output.height = 0;
      output.row_step = 0;
      return output;
    }
    try
    {
      sensor_msgs::PointCloud2Iterator<float> x(output, "x");
      sensor_msgs::PointCloud2Iterator<float> y(output, "y");
      sensor_msgs::PointCloud2Iterator<float> z(output, "z");
      for (; x != x.end(); ++x, ++y, ++z)
      {
        sm::Pose point;
        point.time = message.header.stamp.toSec();
        point.x = *x;
        point.y = *y;
        point.z = *z;
        const sm::Pose aligned = selector_.alignedCandidatePose(point);
        *x = static_cast<float>(aligned.x);
        *y = static_cast<float>(aligned.y);
        *z = static_cast<float>(aligned.z);
      }
    }
    catch (const std::runtime_error &error)
    {
      ROS_ERROR_STREAM_THROTTLE(
          2.0, "[ ShadowMux ] cannot align candidate cloud: " << error.what());
      if (selector_.positionBridgeEnabled())
      {
        output.data.clear();
        output.width = 0;
        output.height = 0;
        output.row_step = 0;
      }
    }
    return output;
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  bool rearm_after_internal_rejection_ = false;
  bool frame_epoch_diagnostics_enabled_ = false;
  bool frame_epoch_snapshot_preview_enabled_ = false;
  bool source_revision_cut_request_enabled_ = false;
  bool source_revision_barrier_enabled_ = false;
  bool source_revision_payload_hash_enabled_ = false;
  bool source_revision_barrier_pending_ = false;
  double source_revision_barrier_timeout_sec_ = 2.0;
  double source_revision_barrier_request_wall_ = 0.0;
  std::string source_revision_barrier_manifest_;
  nav_msgs::Odometry::ConstPtr held_candidate_odom_;
  bool public_candidate_active_ = false;
  std::string public_bridge_frame_id_;
  std::string native_frame_id_;
  std::size_t candidate_epoch_counter_ = 0;
  sm::Selector selector_;
  double cache_horizon_sec_ = 180.0;
  double latest_published_incumbent_time_ = -1.0;
  double last_active_odom_time_ = -1.0;
  double published_odom_watermark_ = -1.0;
  double last_active_path_time_ = -1.0;
  double last_active_cloud_time_ = -1.0;
  double last_active_map_time_ = -1.0;
  double switch_time_ = -1.0;
  double first_public_boundary_raw_position_m_ =
      std::numeric_limits<double>::quiet_NaN();
  double first_public_boundary_corrected_position_m_ =
      std::numeric_limits<double>::quiet_NaN();
  std::size_t dropped_nonmonotonic_odom_ = 0;
  std::size_t candidate_callback_sequence_ = 0;
  double latest_candidate_callback_time_ =
      -std::numeric_limits<double>::infinity();
  double maximum_candidate_time_ = -std::numeric_limits<double>::infinity();
  double commit_status_sensor_time_sec_ =
      std::numeric_limits<double>::quiet_NaN();
  double commit_status_relative_time_sec_ =
      std::numeric_limits<double>::quiet_NaN();
  std::string boundary_diagnostics_path_;
  std::string callback_trace_path_;
  std::string candidate_validation_detail_;
  bool awaiting_internal_rearm_ = false;
  std::ofstream boundary_diagnostics_;
  std::ofstream callback_trace_;
  std::size_t callback_trace_index_ = 0;
  std::map<double, nav_msgs::Odometry::ConstPtr> incumbent_odom_cache_;
  nav_msgs::Odometry::ConstPtr pending_incumbent_odom_;
  nav_msgs::Path::ConstPtr pending_incumbent_path_;
  nav_msgs::Path::ConstPtr cached_candidate_path_;
  sensor_msgs::PointCloud2::ConstPtr pending_incumbent_cloud_;
  sensor_msgs::PointCloud2::ConstPtr pending_incumbent_map_;

  ros::Subscriber incumbent_odom_sub_, candidate_odom_sub_;
  ros::Subscriber incumbent_path_sub_, candidate_path_sub_;
  ros::Subscriber incumbent_cloud_sub_, candidate_cloud_sub_;
  ros::Subscriber incumbent_map_sub_, candidate_map_sub_;
  ros::Subscriber candidate_status_sub_;
  ros::Subscriber finalize_sub_;
  ros::Subscriber source_revision_ready_sub_;
  ros::Publisher active_odom_pub_, active_path_pub_, active_cloud_pub_, active_map_pub_;
  ros::Publisher generation_pub_, abort_pub_;
  ros::Publisher native_odom_pub_, native_path_pub_, native_cloud_pub_, native_map_pub_;
  ros::Publisher frame_epoch_contract_pub_;
  ros::Publisher source_revision_cut_request_pub_;
  ros::Publisher source_revision_manifest_pub_;
  ros::WallTimer source_revision_barrier_timer_;
  ros::Publisher frame_epoch_versioned_manifest_pub_;
  ros::Publisher snapshot_preview_path_pub_, snapshot_preview_map_pub_;
  ros::Publisher snapshot_preview_bundle_pub_;
  tf::TransformBroadcaster transform_broadcaster_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "shadow_output_mux");
  try
  {
    ShadowOutputMux mux;
    ros::spin();
  }
  catch (const std::exception &error)
  {
    ROS_FATAL_STREAM("[ ShadowMux ] " << error.what());
    return 1;
  }
  return 0;
}
