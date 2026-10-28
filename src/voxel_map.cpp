/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"
#include "checkpoint_binary_codec.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
using VoxelMapStorage = std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *>;
using fast_livo::checkpoint_io::Reader;
using fast_livo::checkpoint_io::Writer;

constexpr std::uint64_t kVoxelSnapshotMagic = 0x3150414d584f5646ULL; // FVOXMAP1
constexpr std::uint32_t kVoxelSnapshotSchemaVersion = 1;

void writePointWithVar(Writer &writer, const pointWithVar &point)
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

pointWithVar readPointWithVar(Reader &reader)
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

void writeVoxelPlane(Writer &writer, const VoxelPlane &plane)
{
  writer.eigen(plane.center_);
  writer.eigen(plane.normal_);
  writer.eigen(plane.y_normal_);
  writer.eigen(plane.x_normal_);
  writer.eigen(plane.covariance_);
  writer.eigen(plane.plane_var_);
  writer.pod(plane.radius_);
  writer.pod(plane.min_eigen_value_);
  writer.pod(plane.mid_eigen_value_);
  writer.pod(plane.max_eigen_value_);
  writer.pod(plane.d_);
  writer.pod(plane.points_size_);
  writer.pod(static_cast<std::uint8_t>(plane.is_plane_));
  writer.pod(static_cast<std::uint8_t>(plane.is_init_));
  writer.pod(plane.id_);
  writer.pod(static_cast<std::uint8_t>(plane.is_update_));
}

VoxelPlane readVoxelPlane(Reader &reader)
{
  VoxelPlane plane;
  reader.eigen(plane.center_);
  reader.eigen(plane.normal_);
  reader.eigen(plane.y_normal_);
  reader.eigen(plane.x_normal_);
  reader.eigen(plane.covariance_);
  reader.eigen(plane.plane_var_);
  plane.radius_ = reader.pod<float>();
  plane.min_eigen_value_ = reader.pod<float>();
  plane.mid_eigen_value_ = reader.pod<float>();
  plane.max_eigen_value_ = reader.pod<float>();
  plane.d_ = reader.pod<float>();
  plane.points_size_ = reader.pod<int>();
  plane.is_plane_ = reader.pod<std::uint8_t>() != 0;
  plane.is_init_ = reader.pod<std::uint8_t>() != 0;
  plane.id_ = reader.pod<int>();
  plane.is_update_ = reader.pod<std::uint8_t>() != 0;
  return plane;
}

void writeVoxelNode(Writer &writer, const VoxelOctoTree *node)
{
  writer.pod(static_cast<std::uint8_t>(node != nullptr));
  if (node == nullptr) return;
  writer.pod(node->layer_);
  writer.pod(node->octo_state_);
  writer.pod(node->quater_length_);
  writer.pod(node->planer_threshold_);
  writer.pod(node->points_size_threshold_);
  writer.pod(node->update_size_threshold_);
  writer.pod(node->max_points_num_);
  writer.pod(node->max_layer_);
  writer.pod(node->new_points_);
  writer.pod(static_cast<std::uint8_t>(node->init_octo_));
  writer.pod(static_cast<std::uint8_t>(node->update_enable_));
  for (const double value : node->voxel_center_) writer.pod(value);
  writer.pod(static_cast<std::uint64_t>(node->layer_init_num_.size()));
  for (const int value : node->layer_init_num_) writer.pod(value);
  writer.pod(static_cast<std::uint64_t>(node->temp_points_.size()));
  for (const pointWithVar &point : node->temp_points_) writePointWithVar(writer, point);
  writer.pod(static_cast<std::uint8_t>(node->plane_ptr_ != nullptr));
  if (node->plane_ptr_ != nullptr) writeVoxelPlane(writer, *node->plane_ptr_);
  for (const VoxelOctoTree *child : node->leaves_) writeVoxelNode(writer, child);
}

VoxelOctoTree *readVoxelNode(Reader &reader, const int depth = 0)
{
  if (reader.pod<std::uint8_t>() == 0) return nullptr;
  const std::size_t node_start = reader.cursor() - 1;
  try
  {
  const int layer = reader.pod<int>();
  const int octo_state = reader.pod<int>();
  const float quater_length = reader.pod<float>();
  const float planer_threshold = reader.pod<float>();
  const int points_size_threshold = reader.pod<int>();
  const int update_size_threshold = reader.pod<int>();
  const int max_points_num = reader.pod<int>();
  const int max_layer = reader.pod<int>();
  const int new_points = reader.pod<int>();
  const bool init_octo = reader.pod<std::uint8_t>() != 0;
  const bool update_enable = reader.pod<std::uint8_t>() != 0;
  std::unique_ptr<VoxelOctoTree> node(new VoxelOctoTree(
      max_layer, layer, points_size_threshold, max_points_num,
      planer_threshold));
  node->octo_state_ = octo_state;
  node->quater_length_ = quater_length;
  node->update_size_threshold_ = update_size_threshold;
  node->new_points_ = new_points;
  node->init_octo_ = init_octo;
  node->update_enable_ = update_enable;
  for (double &value : node->voxel_center_) value = reader.pod<double>();
  const std::uint64_t layer_count = fast_livo::checkpoint_io::checkedCount(
      reader, 1024, "octree layer-initialization");
  node->layer_init_num_.resize(static_cast<std::size_t>(layer_count));
  for (int &value : node->layer_init_num_) value = reader.pod<int>();
  const std::uint64_t point_count = fast_livo::checkpoint_io::checkedCount(
      reader, 10000000, "octree point");
  node->temp_points_.reserve(static_cast<std::size_t>(point_count));
  for (std::uint64_t index = 0; index < point_count; ++index)
    node->temp_points_.push_back(readPointWithVar(reader));
  const bool has_plane = reader.pod<std::uint8_t>() != 0;
  if (has_plane)
    *node->plane_ptr_ = readVoxelPlane(reader);
  else
  {
    delete node->plane_ptr_;
    node->plane_ptr_ = nullptr;
  }
  for (int child = 0; child < 8; ++child)
    node->leaves_[child] = readVoxelNode(reader, depth + 1);
  return node.release();
  }
  catch (const std::exception &error)
  {
    throw std::runtime_error(
        "voxel node decode failed at depth " + std::to_string(depth) +
        ", byte " + std::to_string(node_start) + ": " + error.what());
  }
}

void deleteVoxelMap(VoxelMapStorage &voxel_map)
{
  for (auto &entry : voxel_map) delete entry.second;
  voxel_map.clear();
}

VoxelMapStorage cloneVoxelMap(const VoxelMapStorage &source)
{
  VoxelMapStorage clone;
  clone.reserve(source.size());
  try
  {
    for (const auto &entry : source)
    {
      clone.emplace(entry.first, entry.second == nullptr ? nullptr : new VoxelOctoTree(*entry.second));
    }
  }
  catch (...)
  {
    deleteVoxelMap(clone);
    throw;
  }
  return clone;
}

bool finiteVector(const Eigen::Vector3d &value)
{
  return std::isfinite(value.x()) && std::isfinite(value.y()) &&
      std::isfinite(value.z());
}

bool findPointMutationTarget(const VoxelOctoTree *node,
                             const pointWithVar &point,
                             std::vector<int> &path,
                             const VoxelOctoTree *&target)
{
  if (node == nullptr)
  {
    target = nullptr;
    return true;
  }
  if (!node->init_octo_)
  {
    target = node;
    return true;
  }
  if (node->plane_ptr_ != nullptr && node->plane_ptr_->is_plane_)
  {
    if (!node->update_enable_) return false;
    target = node;
    return true;
  }
  if (node->layer_ >= node->max_layer_)
  {
    if (!node->update_enable_) return false;
    target = node;
    return true;
  }

  int xyz[3] = {0, 0, 0};
  if (point.point_w[0] > node->voxel_center_[0]) xyz[0] = 1;
  if (point.point_w[1] > node->voxel_center_[1]) xyz[1] = 1;
  if (point.point_w[2] > node->voxel_center_[2]) xyz[2] = 1;
  const int leaf = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
  path.push_back(leaf);
  if (node->leaves_[leaf] == nullptr)
  {
    target = nullptr;
    return true;
  }
  return findPointMutationTarget(node->leaves_[leaf], point, path, target);
}

bool pathIsPrefix(const std::vector<int> &prefix,
                  const std::vector<int> &path)
{
  return prefix.size() <= path.size() &&
      std::equal(prefix.begin(), prefix.end(), path.begin());
}

void appendRepresentativePoint(const Eigen::Vector3d &position,
                               const float intensity,
                               const std::size_t maximum_points,
                               PointCloudXYZI &cloud,
                               VoxelMapManager::RepresentativeMapStatistics &statistics)
{
  if (!finiteVector(position)) return;
  if (cloud.size() >= maximum_points)
  {
    statistics.truncated = true;
    return;
  }
  PointType point{};
  point.x = static_cast<float>(position.x());
  point.y = static_cast<float>(position.y());
  point.z = static_cast<float>(position.z());
  point.intensity = intensity;
  cloud.push_back(point);
}

void countRepresentativeNode(
    const VoxelOctoTree *node,
    VoxelMapManager::RepresentativeMapStatistics &statistics)
{
  if (node == nullptr) return;
  const bool plane = node->plane_ptr_ != nullptr &&
      node->plane_ptr_->is_plane_ &&
      finiteVector(node->plane_ptr_->center_);
  bool has_child = false;
  if (!plane)
  {
    for (const VoxelOctoTree *child : node->leaves_)
    {
      if (child == nullptr) continue;
      has_child = true;
      countRepresentativeNode(child, statistics);
    }
  }
  if (has_child) return;
  if (plane)
  {
    ++statistics.terminal_nodes;
    ++statistics.plane_nodes;
    return;
  }
  bool has_finite_point = false;
  for (const pointWithVar &point : node->temp_points_)
  {
    if (finiteVector(point.point_w))
    {
      has_finite_point = true;
      break;
    }
  }
  if (has_finite_point)
  {
    ++statistics.terminal_nodes;
    ++statistics.fallback_nodes;
  }
}

void exportRepresentativeNode(
    const VoxelOctoTree *node, const std::size_t maximum_points,
    const std::size_t samples_per_plane, PointCloudXYZI &cloud,
    VoxelMapManager::RepresentativeMapStatistics &statistics)
{
  if (node == nullptr || statistics.truncated) return;

  const bool plane = node->plane_ptr_ != nullptr &&
      node->plane_ptr_->is_plane_ &&
      finiteVector(node->plane_ptr_->center_);
  bool has_child = false;
  if (!plane)
  {
    for (const VoxelOctoTree *child : node->leaves_)
    {
      if (child == nullptr) continue;
      has_child = true;
      exportRepresentativeNode(child, maximum_points, samples_per_plane,
                               cloud, statistics);
      if (statistics.truncated) return;
    }
  }
  if (has_child) return;

  if (plane)
  {
    const VoxelPlane &surface = *node->plane_ptr_;
    const Eigen::Vector3d center = surface.center_;
    Eigen::Vector3d x_axis = surface.x_normal_;
    Eigen::Vector3d y_axis = surface.y_normal_;
    if (!finiteVector(x_axis) || x_axis.norm() < 1e-9 ||
        !finiteVector(y_axis) || y_axis.norm() < 1e-9)
    {
      appendRepresentativePoint(center,
          static_cast<float>(surface.points_size_), maximum_points, cloud,
          statistics);
      return;
    }
    x_axis.normalize();
    y_axis.normalize();
    const double cell_half_extent = std::max(
        0.01, 2.0 * static_cast<double>(node->quater_length_));
    const double x_sigma = std::sqrt(std::max(
        0.0, static_cast<double>(surface.max_eigen_value_)));
    const double y_sigma = std::sqrt(std::max(
        0.0, static_cast<double>(surface.mid_eigen_value_)));
    const double x_extent = std::min(
        cell_half_extent, std::max(0.01, 2.0 * x_sigma));
    const double y_extent = std::min(
        cell_half_extent, std::max(0.01, 2.0 * y_sigma));
    static const std::array<std::array<int, 2>, 9> offsets{{
        {{0, 0}}, {{-1, 0}}, {{1, 0}}, {{0, -1}}, {{0, 1}},
        {{-1, -1}}, {{-1, 1}}, {{1, -1}}, {{1, 1}}
    }};
    const std::size_t count = std::min(samples_per_plane, offsets.size());
    for (std::size_t index = 0; index < count; ++index)
    {
      const Eigen::Vector3d position = center +
          static_cast<double>(offsets[index][0]) * x_extent * x_axis +
          static_cast<double>(offsets[index][1]) * y_extent * y_axis;
      appendRepresentativePoint(position,
          static_cast<float>(surface.points_size_), maximum_points, cloud,
          statistics);
      if (statistics.truncated) return;
    }
    return;
  }

  if (node->temp_points_.empty()) return;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  std::size_t finite_count = 0;
  for (const pointWithVar &point : node->temp_points_)
  {
    if (!finiteVector(point.point_w)) continue;
    centroid += point.point_w;
    ++finite_count;
  }
  if (finite_count == 0) return;
  centroid /= static_cast<double>(finite_count);
  appendRepresentativePoint(centroid, static_cast<float>(finite_count),
                            maximum_points, cloud, statistics);
}
} // namespace

VoxelOctoTree::VoxelOctoTree(const VoxelOctoTree &other)
    : temp_points_(other.temp_points_),
      plane_ptr_(other.plane_ptr_ == nullptr ? nullptr : new VoxelPlane(*other.plane_ptr_)),
      layer_(other.layer_),
      octo_state_(other.octo_state_),
      layer_init_num_(other.layer_init_num_),
      quater_length_(other.quater_length_),
      planer_threshold_(other.planer_threshold_),
      points_size_threshold_(other.points_size_threshold_),
      update_size_threshold_(other.update_size_threshold_),
      max_points_num_(other.max_points_num_),
      max_layer_(other.max_layer_),
      new_points_(other.new_points_),
      init_octo_(other.init_octo_),
      update_enable_(other.update_enable_)
{
  for (int i = 0; i < 3; ++i) voxel_center_[i] = other.voxel_center_[i];
  for (int i = 0; i < 8; ++i) leaves_[i] = nullptr;
  try
  {
    for (int i = 0; i < 8; ++i)
    {
      leaves_[i] = other.leaves_[i] == nullptr ? nullptr : new VoxelOctoTree(*other.leaves_[i]);
    }
  }
  catch (...)
  {
    for (int i = 0; i < 8; ++i) delete leaves_[i];
    delete plane_ptr_;
    plane_ptr_ = nullptr;
    throw;
  }
}

VoxelMapSnapshot::~VoxelMapSnapshot() { deleteVoxelMap(voxel_map_); }

VoxelMapSnapshot::VoxelMapSnapshot(VoxelMapSnapshot &&other) noexcept
{
  voxel_map_.swap(other.voxel_map_);
}

VoxelMapSnapshot &VoxelMapSnapshot::operator=(VoxelMapSnapshot &&other) noexcept
{
  if (this == &other) return *this;
  deleteVoxelMap(voxel_map_);
  voxel_map_.swap(other.voxel_map_);
  return *this;
}

std::vector<std::uint8_t> VoxelMapSnapshot::serialize() const
{
  Writer writer;
  writer.pod(kVoxelSnapshotMagic);
  writer.pod(kVoxelSnapshotSchemaVersion);
  std::vector<VOXEL_LOCATION> keys;
  keys.reserve(voxel_map_.size());
  for (const auto &entry : voxel_map_) keys.push_back(entry.first);
  std::sort(keys.begin(), keys.end(), [](const VOXEL_LOCATION &left,
                                         const VOXEL_LOCATION &right) {
    if (left.x != right.x) return left.x < right.x;
    if (left.y != right.y) return left.y < right.y;
    return left.z < right.z;
  });
  writer.pod(static_cast<std::uint64_t>(keys.size()));
  for (const VOXEL_LOCATION &key : keys)
  {
    writer.pod(key.x);
    writer.pod(key.y);
    writer.pod(key.z);
    writeVoxelNode(writer, voxel_map_.at(key));
  }
  const std::uint64_t checksum =
      fast_livo::checkpoint_io::fnv1a(writer.bytes());
  writer.pod(checksum);
  return writer.take();
}

std::unique_ptr<VoxelMapSnapshot> VoxelMapSnapshot::deserialize(
    const std::vector<std::uint8_t> &payload)
{
  if (payload.size() < sizeof(std::uint64_t))
    throw std::runtime_error("truncated voxel snapshot checksum");
  std::uint64_t stored_checksum = 0;
  std::memcpy(&stored_checksum,
              payload.data() + payload.size() - sizeof(stored_checksum),
              sizeof(stored_checksum));
  const std::vector<std::uint8_t> body(
      payload.begin(), payload.end() - sizeof(stored_checksum));
  if (fast_livo::checkpoint_io::fnv1a(body) != stored_checksum)
    throw std::runtime_error("voxel snapshot checksum mismatch");
  Reader reader(body);
  if (reader.pod<std::uint64_t>() != kVoxelSnapshotMagic)
    throw std::runtime_error("invalid voxel snapshot magic");
  if (reader.pod<std::uint32_t>() != kVoxelSnapshotSchemaVersion)
    throw std::runtime_error("unsupported voxel snapshot schema");
  const std::uint64_t root_count = fast_livo::checkpoint_io::checkedCount(
      reader, 100000000, "root voxel");
  std::unique_ptr<VoxelMapSnapshot> snapshot(new VoxelMapSnapshot());
  snapshot->voxel_map_.reserve(static_cast<std::size_t>(root_count));
  try
  {
    for (std::uint64_t index = 0; index < root_count; ++index)
    {
      const std::int64_t x = reader.pod<std::int64_t>();
      const std::int64_t y = reader.pod<std::int64_t>();
      const std::int64_t z = reader.pod<std::int64_t>();
      const VOXEL_LOCATION key(x, y, z);
      VoxelOctoTree *node = readVoxelNode(reader);
      const auto inserted = snapshot->voxel_map_.emplace(key, node);
      if (!inserted.second)
      {
        delete node;
        throw std::runtime_error("duplicate root voxel in checkpoint");
      }
    }
    if (!reader.finished())
      throw std::runtime_error("trailing bytes in voxel snapshot");
  }
  catch (...)
  {
    snapshot.reset();
    throw;
  }
  return snapshot;
}

std::uint64_t VoxelMapSnapshot::fingerprint() const
{
  return fast_livo::checkpoint_io::fnv1a(serialize());
}

std::unique_ptr<VoxelMapSnapshot> VoxelMapManager::CreateSnapshot() const
{
  std::unique_ptr<VoxelMapSnapshot> snapshot(new VoxelMapSnapshot());
  snapshot->voxel_map_ = cloneVoxelMap(voxel_map_);
  return snapshot;
}

void VoxelMapManager::RestoreSnapshot(const VoxelMapSnapshot &snapshot)
{
  // The transactional mapper intentionally keeps the verified snapshot alive
  // until commit so it can reconstruct the incumbent after a rejected replay.
  // Cloning before releasing the polluted live map transiently holds three
  // complete maps and can exceed memory on long sequences. Release the
  // disposable live generation first; the snapshot remains immutable and can
  // still be reused by a later rejection rebuild.
  deleteVoxelMap(voxel_map_);
  voxel_map_ = cloneVoxelMap(snapshot.voxel_map_);
}

void VoxelMapManager::ConsumeSnapshot(VoxelMapSnapshot &snapshot)
{
  // A process-isolated shadow candidate never needs to reconstruct the
  // separately running incumbent after rejection. Transfer ownership of the
  // verified generation into the live map without allocating a second copy.
  deleteVoxelMap(voxel_map_);
  voxel_map_.swap(snapshot.voxel_map_);
}

void VoxelMapManager::ClearVoxelMap() { deleteVoxelMap(voxel_map_); }

std::size_t VoxelMapManager::BeginMutationJournal(
    const MutationRedoImage *restore_coverage)
{
  MutationJournal journal;
  journal.id = next_mutation_journal_id_++;
  if (restore_coverage != nullptr)
  {
    for (const auto &entry : restore_coverage->root_after_images)
      journal.overwrite_roots.emplace(entry.first, true);
    for (const auto &entry : restore_coverage->node_after_images)
    {
      std::vector<std::vector<int>> &paths =
          journal.overwrite_node_paths[entry.first];
      paths.reserve(entry.second.size());
      for (const MutationRedoImage::NodeAfterImage &image : entry.second)
        paths.push_back(image.path);
    }
  }
  mutation_journals_.push_back(std::move(journal));
  return mutation_journals_.back().id;
}

VoxelMapManager::MutationJournal *VoxelMapManager::FindMutationJournal(
    const std::size_t journal_id)
{
  if (mutation_journals_.empty()) return nullptr;
  if (journal_id == 0) return &mutation_journals_.front();
  for (MutationJournal &journal : mutation_journals_)
    if (journal.id == journal_id) return &journal;
  return nullptr;
}

const VoxelMapManager::MutationJournal *VoxelMapManager::FindMutationJournal(
    const std::size_t journal_id) const
{
  if (mutation_journals_.empty()) return nullptr;
  if (journal_id == 0) return &mutation_journals_.front();
  for (const MutationJournal &journal : mutation_journals_)
    if (journal.id == journal_id) return &journal;
  return nullptr;
}

void VoxelMapManager::RecordMutation(const VOXEL_LOCATION &position)
{
  const auto live = voxel_map_.find(position);
  for (MutationJournal &journal : mutation_journals_)
  {
    if (journal.overwrite_roots.find(position) !=
        journal.overwrite_roots.end())
      continue;
    if (journal.before_images.find(position) != journal.before_images.end() ||
        journal.inserted_keys.find(position) != journal.inserted_keys.end())
      continue;
    if (live == voxel_map_.end())
    {
      journal.inserted_keys.emplace(position, true);
      ++journal.dirty_count;
    }
    else
    {
      const double clone_start = omp_get_wtime();
      std::unique_ptr<VoxelOctoTree> before(
          live->second == nullptr ? nullptr : new VoxelOctoTree(*live->second));
      journal.clone_wall_sec += omp_get_wtime() - clone_start;
      journal.before_images.emplace(position, std::move(before));
      ++journal.dirty_count;
    }
    journal.peak_dirty_roots = std::max(
        journal.peak_dirty_roots, journal.dirty_count);
  }
}

void VoxelMapManager::RecordPointMutation(
    const VOXEL_LOCATION &position, const pointWithVar &point)
{
  if (mutation_journals_.empty()) return;
  const auto live = voxel_map_.find(position);
  if (live == voxel_map_.end())
  {
    RecordMutation(position);
    return;
  }

  std::vector<int> path;
  const VoxelOctoTree *target = nullptr;
  if (!findPointMutationTarget(live->second, point, path, target)) return;

  for (MutationJournal &journal : mutation_journals_)
  {
    if (journal.overwrite_roots.find(position) !=
        journal.overwrite_roots.end())
      continue;
    const auto overwrite_paths = journal.overwrite_node_paths.find(position);
    if (overwrite_paths != journal.overwrite_node_paths.end())
    {
      bool overwritten_by_redo = false;
      for (const std::vector<int> &overwrite_path : overwrite_paths->second)
      {
        if (pathIsPrefix(overwrite_path, path))
        {
          overwritten_by_redo = true;
          break;
        }
      }
      if (overwritten_by_redo) continue;
    }
    if (journal.inserted_keys.find(position) != journal.inserted_keys.end() ||
        journal.before_images.find(position) != journal.before_images.end())
      continue;
    std::vector<NodeBeforeImage> &images = journal.node_before_images[position];
    bool covered = false;
    for (const NodeBeforeImage &image : images)
    {
      if (pathIsPrefix(image.path, path))
      {
        covered = true;
        break;
      }
    }
    if (covered) continue;

    const double clone_start = omp_get_wtime();
    NodeBeforeImage image;
    image.path = path;
    if (target != nullptr) image.node.reset(new VoxelOctoTree(*target));
    journal.clone_wall_sec += omp_get_wtime() - clone_start;
    images.push_back(std::move(image));
    ++journal.dirty_count;
    journal.peak_dirty_roots = std::max(
        journal.peak_dirty_roots, journal.dirty_count);
  }
}

void VoxelMapManager::RollbackMutationJournal(const std::size_t journal_id)
{
  MutationJournal *selected = FindMutationJournal(journal_id);
  if (selected == nullptr) return;

  for (const auto &inserted : selected->inserted_keys)
  {
    const auto live = voxel_map_.find(inserted.first);
    if (live != voxel_map_.end())
    {
      delete live->second;
      voxel_map_.erase(live);
    }
  }
  for (auto &before : selected->before_images)
  {
    const auto live = voxel_map_.find(before.first);
    if (live != voxel_map_.end())
    {
      delete live->second;
      live->second = before.second.release();
    }
    else
    {
      voxel_map_.emplace(before.first, before.second.release());
    }
  }
  for (auto &root_images : selected->node_before_images)
  {
    auto root = voxel_map_.find(root_images.first);
    if (root == voxel_map_.end()) continue;
    std::vector<NodeBeforeImage *> ordered;
    ordered.reserve(root_images.second.size());
    for (NodeBeforeImage &image : root_images.second)
      ordered.push_back(&image);
    std::stable_sort(ordered.begin(), ordered.end(),
        [](const NodeBeforeImage *left, const NodeBeforeImage *right) {
          return left->path.size() < right->path.size();
        });
    for (NodeBeforeImage *image : ordered)
    {
      VoxelOctoTree **slot = &root->second;
      bool path_valid = true;
      for (const int child : image->path)
      {
        if (*slot == nullptr || child < 0 || child >= 8)
        {
          path_valid = false;
          break;
        }
        slot = &((*slot)->leaves_[child]);
      }
      if (!path_valid) continue;
      delete *slot;
      *slot = image->node.release();
    }
  }
  mutation_journals_.clear();
}

void VoxelMapManager::CommitMutationJournal()
{
  mutation_journals_.clear();
}

void VoxelMapManager::DiscardMutationJournal(const std::size_t journal_id)
{
  for (auto it = mutation_journals_.begin(); it != mutation_journals_.end(); ++it)
  {
    if (it->id != journal_id) continue;
    mutation_journals_.erase(it);
    return;
  }
}

bool VoxelMapManager::MutationJournalActive(const std::size_t journal_id) const
{
  return FindMutationJournal(journal_id) != nullptr;
}

std::size_t VoxelMapManager::MutationJournalBeforeImageCount(
    const std::size_t journal_id) const
{
  const MutationJournal *journal = FindMutationJournal(journal_id);
  if (journal == nullptr) return 0;
  std::size_t count = journal->before_images.size();
  for (const auto &entry : journal->node_before_images)
    count += entry.second.size();
  return count;
}

std::size_t VoxelMapManager::MutationJournalInsertedCount(
    const std::size_t journal_id) const
{
  const MutationJournal *journal = FindMutationJournal(journal_id);
  return journal == nullptr ? 0 : journal->inserted_keys.size();
}

std::size_t VoxelMapManager::MutationJournalDirtyCount(
    const std::size_t journal_id) const
{
  return MutationJournalBeforeImageCount(journal_id) +
      MutationJournalInsertedCount(journal_id);
}

std::size_t VoxelMapManager::MutationJournalPeakDirtyCount(
    const std::size_t journal_id) const
{
  const MutationJournal *journal = FindMutationJournal(journal_id);
  return journal == nullptr ? 0 : journal->peak_dirty_roots;
}

double VoxelMapManager::MutationJournalCloneWallSec(
    const std::size_t journal_id) const
{
  const MutationJournal *journal = FindMutationJournal(journal_id);
  return journal == nullptr ? 0.0 : journal->clone_wall_sec;
}

std::unique_ptr<VoxelMapManager::MutationRedoImage>
VoxelMapManager::CaptureMutationRedo(const std::size_t journal_id) const
{
  const MutationJournal *journal = FindMutationJournal(journal_id);
  if (journal == nullptr) return nullptr;

  std::unique_ptr<MutationRedoImage> redo(new MutationRedoImage());
  redo->source_journal_id = journal->id;
  const double clone_start = omp_get_wtime();

  const auto capture_root = [this, &redo](const VOXEL_LOCATION &key) {
    if (redo->root_after_images.find(key) != redo->root_after_images.end())
      return;
    const auto live = voxel_map_.find(key);
    redo->root_after_images.emplace(
        key, std::unique_ptr<VoxelOctoTree>(
                 live == voxel_map_.end() || live->second == nullptr ?
                     nullptr : new VoxelOctoTree(*live->second)));
  };
  for (const auto &entry : journal->before_images) capture_root(entry.first);
  for (const auto &entry : journal->inserted_keys) capture_root(entry.first);

  for (const auto &root_images : journal->node_before_images)
  {
    if (redo->root_after_images.find(root_images.first) !=
        redo->root_after_images.end())
      continue;
    const auto live_root = voxel_map_.find(root_images.first);
    std::vector<MutationRedoImage::NodeAfterImage> &after_images =
        redo->node_after_images[root_images.first];
    after_images.reserve(root_images.second.size());
    for (const NodeBeforeImage &before : root_images.second)
    {
      const VoxelOctoTree *node =
          live_root == voxel_map_.end() ? nullptr : live_root->second;
      for (const int child : before.path)
      {
        if (node == nullptr || child < 0 || child >= 8)
        {
          node = nullptr;
          break;
        }
        node = node->leaves_[child];
      }
      MutationRedoImage::NodeAfterImage after;
      after.path = before.path;
      if (node != nullptr) after.node.reset(new VoxelOctoTree(*node));
      after_images.push_back(std::move(after));
    }
  }
  redo->clone_wall_sec = omp_get_wtime() - clone_start;
  return redo;
}

bool VoxelMapManager::ConsumeMutationRedo(MutationRedoImage &redo)
{
  for (auto &after : redo.root_after_images)
  {
    const auto live = voxel_map_.find(after.first);
    if (live != voxel_map_.end())
    {
      delete live->second;
      if (after.second)
        live->second = after.second.release();
      else
        voxel_map_.erase(live);
    }
    else if (after.second)
    {
      voxel_map_.emplace(after.first, after.second.release());
    }
  }
  for (auto &root_images : redo.node_after_images)
  {
    auto root = voxel_map_.find(root_images.first);
    if (root == voxel_map_.end()) return false;
    std::vector<MutationRedoImage::NodeAfterImage *> ordered;
    ordered.reserve(root_images.second.size());
    for (MutationRedoImage::NodeAfterImage &image : root_images.second)
      ordered.push_back(&image);
    std::stable_sort(ordered.begin(), ordered.end(),
        [](const MutationRedoImage::NodeAfterImage *left,
           const MutationRedoImage::NodeAfterImage *right) {
          return left->path.size() < right->path.size();
        });
    for (MutationRedoImage::NodeAfterImage *image : ordered)
    {
      VoxelOctoTree **slot = &root->second;
      for (const int child : image->path)
      {
        if (*slot == nullptr || child < 0 || child >= 8) return false;
        slot = &((*slot)->leaves_[child]);
      }
      delete *slot;
      *slot = image->node.release();
    }
  }
  redo.root_after_images.clear();
  redo.node_after_images.clear();
  return true;
}

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov)
{
  if (pb[2] == 0) pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config)
{
  nh.param<bool>("publish/pub_plane_en", voxel_config.is_pub_plane_map_, false);
  
  nh.param<int>("lio/max_layer", voxel_config.max_layer_, 1);
  nh.param<double>("lio/voxel_size", voxel_config.max_voxel_size_, 0.5);
  nh.param<double>("lio/min_eigen_value", voxel_config.planner_threshold_, 0.01);
  nh.param<double>("lio/sigma_num", voxel_config.sigma_num_, 3);
  nh.param<double>("lio/beam_err", voxel_config.beam_err_, 0.02);
  nh.param<double>("lio/dept_err", voxel_config.dept_err_, 0.05);
  nh.param<vector<int>>("lio/layer_init_num", voxel_config.layer_init_num_, vector<int>{5,5,5,5,5});
  nh.param<int>("lio/max_points_num", voxel_config.max_points_num_, 50);
  nh.param<int>("lio/max_iterations", voxel_config.max_iterations_, 5);

  nh.param<bool>("local_map/map_sliding_en", voxel_config.map_sliding_en, false);
  nh.param<int>("local_map/half_map_size", voxel_config.half_map_size, 100);
  nh.param<double>("local_map/sliding_thresh", voxel_config.sliding_thresh, 8);
}

void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane)
{
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points)
  {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();
    plane->center_ += pv.point_w;
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  if (evalsReal(evalsMin) < planer_threshold_)
  {
    for (int i = 0; i < points.size(); i++)
    {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++)
      {
        if (m != (int)evalsMin)
        {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() / ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        }
        else
        {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_)
    {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  }
  else
  {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

void VoxelOctoTree::init_octo_tree()
{
  if (temp_points_.size() > points_size_threshold_)
  {
    init_plane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true)
    {
      octo_state_ = 0;
      // new added
      if (temp_points_.size() > max_points_num_)
      {
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    }
    else
    {
      octo_state_ = 1;
      cut_octo_tree();
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::cut_octo_tree()
{
  if (layer_ >= max_layer_)
  {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cut_octo_tree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv)
{
  if (!init_octo_)
  {
    new_points_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > points_size_threshold_) { init_octo_tree(); }
  }
  else
  {
    if (plane_ptr_->is_plane_)
    {
      if (update_enable_)
      {
        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_)
        {
          init_plane(temp_points_, plane_ptr_);
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_)
        {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    }
    else
    {
      if (layer_ < max_layer_)
      {
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateOctoTree(pv); }
        else
        {
          leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      }
      else
      {
        if (update_enable_)
        {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_)
          {
            init_plane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_)
          {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw)
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw) : this;
}

VoxelOctoTree *VoxelOctoTree::Insert(const pointWithVar &pv)
{
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_)))
  {
    new_points_++;
    temp_points_.push_back(pv);
    return this;
  }

  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_))
  {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) { return leaves_[leafnum]->Insert(pv); }
    else
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      return leaves_[leafnum]->Insert(pv);
    }
  }
  return nullptr;
}

void VoxelMapManager::StateEstimation(StatesGroup &state_propagat)
{
  cross_mat_list_.clear();
  cross_mat_list_.reserve(feats_down_size_);
  body_cov_list_.clear();
  body_cov_list_.reserve(feats_down_size_);

  // build_residual_time = 0.0;
  // ekf_time = 0.0;
  // double t0 = omp_get_wtime();

  for (size_t i = 0; i < feats_down_body_->size(); i++)
  {
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    if (point_this[2] == 0) { point_this[2] = 0.001; }
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    body_cov_list_.push_back(var);
    point_this = extR_ * point_this + extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    cross_mat_list_.push_back(point_crossmat);
  }

  vector<pointWithVar>().swap(pv_list_);
  pv_list_.resize(feats_down_size_);

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();

  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++)
  {
    double total_residual = 0.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
    TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);
    M3D rot_var = state_.cov.block<3, 3>(0, 0);
    M3D t_var = state_.cov.block<3, 3>(3, 3);
    for (size_t i = 0; i < feats_down_body_->size(); i++)
    {
      pointWithVar &pv = pv_list_[i];
      pv.point_b << feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z;
      pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;

      M3D cov = body_cov_list_[i];
      M3D point_crossmat = cross_mat_list_[i];
      cov = state_.rot_end * cov * state_.rot_end.transpose() + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
      pv.var = cov;
      pv.body_var = body_cov_list_[i];
    }
    ptpl_list_.clear();

    // double t1 = omp_get_wtime();

    BuildResidualListOMP(pv_list_, ptpl_list_);

    // build_residual_time += omp_get_wtime() - t1;

    double raw_total_residual = 0.0;
    std::vector<double> absolute_residuals;
    absolute_residuals.reserve(ptpl_list_.size());
    for (int i = 0; i < ptpl_list_.size(); i++)
    {
      const double residual = fabs(ptpl_list_[i].dis_to_plane_);
      raw_total_residual += residual;
      absolute_residuals.push_back(residual);
    }
    raw_mean_abs_residual_ = ptpl_list_.empty() ?
        std::numeric_limits<double>::infinity() :
        raw_total_residual / static_cast<double>(ptpl_list_.size());
    raw_match_ratio_ = feats_down_size_ > 0 ?
        static_cast<double>(ptpl_list_.size()) / static_cast<double>(feats_down_size_) : 0.0;
    selective_inlier_ratio_ = 1.0;
    selective_gate_threshold_m_ = 0.0;
    robust_weight_mean_ = 1.0;
    robust_weight_minimum_ = 1.0;
    robust_weight_transition_m_ = 0.0;
    if (selective_residual_gate_active_ && !ptpl_list_.empty())
    {
      const auto gate = fast_livo::selective_gating::select(
          absolute_residuals, selective_gate_config_);
      std::vector<PointToPlane> selected;
      selected.reserve(gate.retained_count);
      for (std::size_t index = 0; index < ptpl_list_.size(); ++index)
      {
        if (gate.keep[index]) selected.push_back(ptpl_list_[index]);
      }
      ptpl_list_.swap(selected);
      selective_inlier_ratio_ = gate.retained_fraction;
      selective_gate_threshold_m_ = gate.threshold_m;
    }
    std::vector<double> robust_information_weights(ptpl_list_.size(), 1.0);
    if (robust_residual_weight_active_ && !ptpl_list_.empty())
    {
      std::vector<double> selected_residuals;
      selected_residuals.reserve(ptpl_list_.size());
      for (const auto &ptpl : ptpl_list_)
        selected_residuals.push_back(std::abs(ptpl.dis_to_plane_));
      const auto weighting = fast_livo::robust_weighting::compute(
          selected_residuals, robust_weight_config_);
      robust_information_weights = weighting.weights;
      robust_weight_mean_ = weighting.mean_weight;
      robust_weight_minimum_ = weighting.minimum_applied_weight;
      robust_weight_transition_m_ = weighting.transition_m;
    }
    for (int i = 0; i < ptpl_list_.size(); i++)
    {
      total_residual += fabs(ptpl_list_[i].dis_to_plane_);
    }
    effct_feat_num_ = ptpl_list_.size();
    mean_abs_residual_ = effct_feat_num_ > 0 ? total_residual / effct_feat_num_ :
        std::numeric_limits<double>::infinity();
    effective_match_ratio_ = feats_down_size_ > 0 ?
        static_cast<double>(effct_feat_num_) / static_cast<double>(feats_down_size_) : 0.0;
    cout << "[ LIO ] Raw feature num: " << feats_undistort_->size() << ", downsampled feature num:" << feats_down_size_ 
         << " effective feature num: " << effct_feat_num_ << " average residual: " << mean_abs_residual_
         << " raw residual: " << raw_mean_abs_residual_
         << " selective retention: " << selective_inlier_ratio_
         << " robust mean weight: " << robust_weight_mean_ << endl;

    /*** Computation of Measuremnt Jacobian matrix H and measurents covarience
     * ***/
    MatrixXd Hsub(effct_feat_num_, 6);
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_);
    VectorXd R_inv(effct_feat_num_);
    VectorXd meas_vec(effct_feat_num_);
    meas_vec.setZero();
    for (int i = 0; i < effct_feat_num_; i++)
    {
      auto &ptpl = ptpl_list_[i];
      V3D point_this(ptpl.point_b_);
      point_this = extR_ * point_this + extT_;
      V3D point_body(ptpl.point_b_);
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);

      /*** get the normal vector of closest surface/corner ***/

      V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point_world - ptpl_list_[i].center_;
      J_nq.block<1, 3>(0, 3) = -ptpl_list_[i].normal_;

      M3D var;
      // V3D normal_b = state_.rot_end.inverse() * ptpl_list_[i].normal_;
      // V3D point_b = ptpl_list_[i].point_b_;
      // double cos_theta = fabs(normal_b.dot(point_b) / point_b.norm());
      // ptpl_list_[i].body_cov_ = ptpl_list_[i].body_cov_ * (1.0 / cos_theta) * (1.0 / cos_theta);

      // point_w cov
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) + (-point_crossmat) * state_propagat.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose();

      // point_w cov (another_version)
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) - point_crossmat * state_propagat.cov.block<3, 3>(0, 0) * point_crossmat;

      // point_body cov
      var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose();

      double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();

      R_inv(i) = robust_information_weights[i] /
          (0.001 + sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);
      // R_inv(i) = 1.0 / (sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);

      /*** calculate the Measuremnt Jacobian matrix H ***/
      V3D A(point_crossmat * state_.rot_end.transpose() * ptpl_list_[i].normal_);
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i), ptpl_list_[i].normal_[0] * R_inv(i),
          ptpl_list_[i].normal_[1] * R_inv(i), ptpl_list_[i].normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
    }
    EKF_stop_flg = false;
    flg_EKF_converged = false;
    /*** Iterative Kalman Filter Update ***/
    MatrixXd K(DIM_STATE, effct_feat_num_);
    // auto &&Hsub_T = Hsub.transpose();
    auto &&HTz = Hsub_T_R_inv * meas_vec;
    // fout_dbg<<"HTz: "<<HTz<<endl;
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
    // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6,6>(0,0));
    MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) + state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse()).inverse();
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    auto vec = state_propagat - state_;
    VD(DIM_STATE)
    solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0) - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
    int minRow, minCol;
    state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) { flg_EKF_converged = true; }
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    /*** Rematch Judgement ***/

    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2)))) { rematch_num++; }

    /*** Convergence Judgements and Covariance Update ***/
    if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1)))
    {
      /*** Covariance Update ***/
      // _state.cov = (I_STATE - G) * _state.cov;
      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
          (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) - G.block<DIM_STATE, DIM_STATE>(0, 0)) * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);
      // total_distance += (_state.pos_end - position_last).norm();
      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));

      // VD(DIM_STATE) K_sum  = K.rowwise().sum();
      // VD(DIM_STATE) P_diag = _state.cov.diagonal();
      EKF_stop_flg = true;
    }
    if (EKF_stop_flg) break;
  }

  // double t2 = omp_get_wtime();
  // scan_count++;
  // ekf_time = t2 - t0 - build_residual_time;

  // ave_build_residual_time = ave_build_residual_time * (scan_count - 1) / scan_count + build_residual_time / scan_count;
  // ave_ekf_time = ave_ekf_time * (scan_count - 1) / scan_count + ekf_time / scan_count;

  // cout << "[ Mapping ] ekf_time: " << ekf_time << "s, build_residual_time: " << build_residual_time << "s" << endl;
  // cout << "[ Mapping ] ave_ekf_time: " << ave_ekf_time << "s, ave_build_residual_time: " << ave_build_residual_time << "s" << endl;
}

void VoxelMapManager::TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud)
{
  pcl::PointCloud<pcl::PointXYZI>().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR_ * p + extT_) + t);
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void VoxelMapManager::BuildVoxelMap()
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;

  std::vector<pointWithVar> input_points;

  for (size_t i = 0; i < feats_down_world_->size(); i++)
  {
    pointWithVar pv;
    pv.point_w << feats_down_world_->points[i].x, feats_down_world_->points[i].y, feats_down_world_->points[i].z;
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    var = (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose() +
          (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + state_.cov.block<3, 3>(3, 3);
    pv.var = var;
    input_points.push_back(pv);
  }

  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    RecordMutation(position);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
    }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
    }
  }
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter)
  {
    RecordMutation(iter->first);
    iter->second->init_octo_tree();
  }
}

V3F VoxelMapManager::RGBFromVoxel(const V3D &input_point)
{
  int64_t loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = floor(input_point[j] / config_setting_.max_voxel_size_);
  }

  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  int64_t ind = loc_xyz[0] + loc_xyz[1] + loc_xyz[2];
  uint k((ind + 100000) % 3);
  V3F RGB((k == 0) * 255.0, (k == 1) * 255.0, (k == 2) * 255.0);
  // cout<<"RGB: "<<RGB.transpose()<<endl;
  return RGB;
}

void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points)
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      // Mature planar leaves ignore further samples. Recording a before-image
      // for those no-op updates deep-copied the entire root once per active
      // rolling checkpoint, dominating memory on long sequences.
      RecordPointMutation(position, p_v);
      iter->second->UpdateOctoTree(p_v);
    }
    else
    {
      RecordMutation(position);
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->UpdateOctoTree(p_v);
    }
  }
}

void VoxelMapManager::BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list)
{
  BuildResidualListAgainstMapOMP(pv_list, ptpl_list, voxel_map_);
}

void VoxelMapManager::BuildResidualListAgainstMapOMP(
    std::vector<pointWithVar> &pv_list,
    std::vector<PointToPlane> &ptpl_list,
    const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &map) const
{
  int max_layer = config_setting_.max_layer_;
  double voxel_size = config_setting_.max_voxel_size_;
  double sigma_num = config_setting_.sigma_num_;
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<PointToPlane> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
    useful_ptpl[i] = false;
  }
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < index.size(); i++)
  {
    pointWithVar &pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pv.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = map.find(position);
    if (iter != map.end())
    {
      VoxelOctoTree *current_octo = iter->second;
      PointToPlane single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      build_single_residual(pv, current_octo, 0, is_sucess, prob, single_ptpl);
      if (!is_sucess)
      {
        VOXEL_LOCATION near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) { near_position.x = near_position.x + 1; }
        else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) { near_position.x = near_position.x - 1; }
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) { near_position.y = near_position.y + 1; }
        else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) { near_position.y = near_position.y - 1; }
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) { near_position.z = near_position.z + 1; }
        else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) { near_position.z = near_position.z - 1; }
        auto iter_near = map.find(near_position);
        if (iter_near != map.end()) { build_single_residual(pv, iter_near->second, 0, is_sucess, prob, single_ptpl); }
      }
      if (is_sucess)
      {
        mylock.lock();
        single_ptpl.source_index_ = static_cast<std::size_t>(i);
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      }
      else
      {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++)
  {
    if (useful_ptpl[i]) { ptpl_list.push_back(all_ptpl_list[i]); }
  }
}

VoxelMapManager::ResidualStatistics
VoxelMapManager::EvaluateResidualsAgainstSnapshot(
    const VoxelMapSnapshot &snapshot, const StatesGroup &state,
    const PointCloudXYZI::Ptr &body_points) const
{
  return EvaluateResidualsAgainstMap(snapshot.voxel_map_, state, body_points);
}

VoxelMapManager::SnapshotPoseRefinement
VoxelMapManager::RefinePoseAgainstSnapshot(
    const VoxelMapSnapshot &snapshot, const StatesGroup &prior,
    const PointCloudXYZI::Ptr &body_points) const
{
  SnapshotPoseRefinement result;
  result.state = prior;
  result.initial = EvaluateResidualsAgainstSnapshot(snapshot, prior, body_points);
  result.final = result.initial;
  if (!body_points || body_points->empty() ||
      result.initial.matched_points < 30) return result;

  // Keep this exploratory update near the IMU-predicted pose. Re-association
  // after every accepted step is essential: a fixed-pose scale bank loses
  // correspondences when corrected points move across voxel/plane boundaries.
  constexpr double kHuberTransitionM = 0.05;
  constexpr double kTranslationPriorSigmaM = 0.10;
  constexpr double kRotationPriorSigmaRad = 0.035;
  constexpr double kMaximumTranslationStepM = 0.05;
  constexpr double kMaximumRotationStepRad = 0.01;
  for (int iteration = 0; iteration < 4; ++iteration)
  {
    std::vector<pointWithVar> points(body_points->size());
    const M3D rotation_covariance = result.state.cov.block<3, 3>(0, 0);
    const M3D translation_covariance = result.state.cov.block<3, 3>(3, 3);
    for (std::size_t index = 0; index < body_points->size(); ++index)
    {
      V3D body(body_points->points[index].x,
               body_points->points[index].y,
               body_points->points[index].z);
      if (body.z() == 0.0) body.z() = 0.001;
      M3D body_covariance;
      calcBodyCov(body, config_setting_.dept_err_,
                  config_setting_.beam_err_, body_covariance);
      const V3D imu = extR_ * body + extT_;
      M3D cross;
      cross << SKEW_SYM_MATRX(imu);
      pointWithVar &point = points[index];
      point.point_b = body;
      point.point_w = result.state.rot_end * imu + result.state.pos_end;
      point.body_var = body_covariance;
      point.var = result.state.rot_end * body_covariance *
              result.state.rot_end.transpose() +
          (-cross) * rotation_covariance * (-cross.transpose()) +
          translation_covariance;
    }
    std::vector<PointToPlane> residuals;
    BuildResidualListAgainstMapOMP(points, residuals, snapshot.voxel_map_);
    if (residuals.size() < 30) break;

    Eigen::Matrix<double, 6, 6> information =
        Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> gradient =
        Eigen::Matrix<double, 6, 1>::Zero();
    for (const PointToPlane &match : residuals)
    {
      const V3D imu = extR_ * match.point_b_ + extT_;
      const V3D rotated = result.state.rot_end * imu;
      Eigen::Matrix<double, 6, 1> jacobian;
      jacobian.head<3>() = rotated.cross(match.normal_);
      jacobian.tail<3>() = match.normal_;
      const double residual = match.dis_to_plane_;
      if (!std::isfinite(residual) || !jacobian.allFinite()) continue;
      const double weight = std::min(
          1.0, kHuberTransitionM / std::max(std::abs(residual), 1e-9));
      information.noalias() += weight * jacobian * jacobian.transpose();
      gradient.noalias() += weight * jacobian * residual;
    }
    Eigen::Matrix<double, 6, 1> prior_offset;
    const Eigen::AngleAxisd rotation_offset(
        result.state.rot_end * prior.rot_end.transpose());
    prior_offset.head<3>() = rotation_offset.angle() * rotation_offset.axis();
    prior_offset.tail<3>() = result.state.pos_end - prior.pos_end;
    Eigen::Matrix<double, 6, 1> prior_weight;
    prior_weight.head<3>().setConstant(
        1.0 / (kRotationPriorSigmaRad * kRotationPriorSigmaRad));
    prior_weight.tail<3>().setConstant(
        1.0 / (kTranslationPriorSigmaM * kTranslationPriorSigmaM));
    information.diagonal() += prior_weight;
    gradient += prior_weight.cwiseProduct(prior_offset);
    Eigen::Matrix<double, 6, 1> step =
        information.ldlt().solve(-gradient);
    if (!step.allFinite()) break;
    const double rotation_norm = step.head<3>().norm();
    const double translation_norm = step.tail<3>().norm();
    if (rotation_norm > kMaximumRotationStepRad)
      step.head<3>() *= kMaximumRotationStepRad / rotation_norm;
    if (translation_norm > kMaximumTranslationStepM)
      step.tail<3>() *= kMaximumTranslationStepM / translation_norm;
    if (step.norm() < 1e-6) break;

    StatesGroup proposal = result.state;
    const double bounded_rotation_norm = step.head<3>().norm();
    if (bounded_rotation_norm > 0.0)
      proposal.rot_end = Eigen::AngleAxisd(
          bounded_rotation_norm, step.head<3>() / bounded_rotation_norm)
          .toRotationMatrix() * proposal.rot_end;
    proposal.pos_end += step.tail<3>();
    const ResidualStatistics candidate = EvaluateResidualsAgainstSnapshot(
        snapshot, proposal, body_points);
    if (!candidate.valid ||
        candidate.matched_points <
            static_cast<std::size_t>(0.8 * result.initial.matched_points) ||
        candidate.mean_abs_residual >= result.final.mean_abs_residual)
      break;
    result.state = proposal;
    result.final = candidate;
    ++result.accepted_steps;
  }
  return result;
}

VoxelMapManager::CommonPlaneStatistics
VoxelMapManager::EvaluateCommonPlanesAgainstSnapshot(
    const VoxelMapSnapshot &snapshot,
    const StatesGroup &reference_state,
    const PointCloudXYZI::Ptr &reference_body_points,
    const StatesGroup &candidate_state,
    const PointCloudXYZI::Ptr &candidate_body_points) const
{
  CommonPlaneStatistics statistics;
  if (!reference_body_points || !candidate_body_points ||
      reference_body_points->empty() ||
      reference_body_points->size() != candidate_body_points->size())
    return statistics;
  statistics.shared_points = reference_body_points->size();
  std::vector<pointWithVar> points(reference_body_points->size());
  const M3D rotation_covariance = reference_state.cov.block<3, 3>(0, 0);
  const M3D translation_covariance = reference_state.cov.block<3, 3>(3, 3);
  for (std::size_t index = 0; index < reference_body_points->size(); ++index)
  {
    V3D body(reference_body_points->points[index].x,
             reference_body_points->points[index].y,
             reference_body_points->points[index].z);
    if (body.z() == 0.0) body.z() = 0.001;
    M3D body_covariance;
    calcBodyCov(body, config_setting_.dept_err_,
                config_setting_.beam_err_, body_covariance);
    const V3D imu = extR_ * body + extT_;
    M3D cross;
    cross << SKEW_SYM_MATRX(imu);
    pointWithVar &point = points[index];
    point.point_b = body;
    point.point_w = reference_state.rot_end * imu +
        reference_state.pos_end;
    point.body_var = body_covariance;
    point.var = reference_state.rot_end * body_covariance *
            reference_state.rot_end.transpose() +
        (-cross) * rotation_covariance * (-cross.transpose()) +
        translation_covariance;
  }
  std::vector<PointToPlane> reference_matches;
  BuildResidualListAgainstMapOMP(
      points, reference_matches, snapshot.voxel_map_);
  statistics.reference_matches = reference_matches.size();
  if (reference_matches.size() < 30) return statistics;

  double reference_total = 0.0;
  double candidate_total = 0.0;
  double displacement_total = 0.0;
  for (const PointToPlane &match : reference_matches)
  {
    const PointType &candidate =
        candidate_body_points->points[match.source_index_];
    const V3D candidate_body(candidate.x, candidate.y, candidate.z);
    const V3D candidate_world = candidate_state.rot_end *
        (extR_ * candidate_body + extT_) + candidate_state.pos_end;
    if (!candidate_world.allFinite()) return statistics;
    reference_total += std::abs(match.dis_to_plane_);
    candidate_total += std::abs(
        match.normal_.dot(candidate_world - match.center_));
    displacement_total += (candidate_world - match.point_w_).norm();
  }
  const double count = static_cast<double>(reference_matches.size());
  statistics.reference_mean_abs_residual = reference_total / count;
  statistics.candidate_mean_abs_residual = candidate_total / count;
  statistics.mean_world_displacement = displacement_total / count;
  statistics.valid = std::isfinite(statistics.reference_mean_abs_residual) &&
      std::isfinite(statistics.candidate_mean_abs_residual) &&
      std::isfinite(statistics.mean_world_displacement);
  return statistics;
}

VoxelMapManager::ResidualStatistics
VoxelMapManager::EvaluateResidualsAgainstCurrentMap(
    const StatesGroup &state,
    const PointCloudXYZI::Ptr &body_points) const
{
  return EvaluateResidualsAgainstMap(voxel_map_, state, body_points);
}

VoxelMapManager::ResidualStatistics
VoxelMapManager::EvaluateResidualsAgainstMap(
    const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &map,
    const StatesGroup &state,
    const PointCloudXYZI::Ptr &body_points) const
{
  ResidualStatistics statistics;
  if (!body_points || body_points->empty()) return statistics;

  statistics.total_points = body_points->size();
  std::vector<pointWithVar> points(body_points->size());
  const M3D rotation_covariance = state.cov.block<3, 3>(0, 0);
  const M3D translation_covariance = state.cov.block<3, 3>(3, 3);
  for (std::size_t index = 0; index < body_points->size(); ++index)
  {
    V3D point_body(body_points->points[index].x,
                   body_points->points[index].y,
                   body_points->points[index].z);
    if (point_body[2] == 0.0) point_body[2] = 0.001;
    M3D body_covariance;
    calcBodyCov(point_body, config_setting_.dept_err_,
                config_setting_.beam_err_, body_covariance);
    const V3D point_imu = extR_ * point_body + extT_;
    M3D point_cross;
    point_cross << SKEW_SYM_MATRX(point_imu);
    pointWithVar &point = points[index];
    point.point_b = point_body;
    point.point_w = state.rot_end * point_imu + state.pos_end;
    point.body_var = body_covariance;
    point.var = state.rot_end * body_covariance * state.rot_end.transpose() +
        (-point_cross) * rotation_covariance * (-point_cross.transpose()) +
        translation_covariance;
  }

  std::vector<PointToPlane> residuals;
  BuildResidualListAgainstMapOMP(points, residuals, map);
  statistics.matched_points = residuals.size();
  statistics.match_ratio = static_cast<double>(statistics.matched_points) /
      static_cast<double>(statistics.total_points);
  double total_residual = 0.0;
  for (const PointToPlane &residual : residuals)
    total_residual += std::abs(residual.dis_to_plane_);
  if (!residuals.empty())
  {
    statistics.mean_abs_residual = total_residual /
        static_cast<double>(residuals.size());
    statistics.valid = std::isfinite(statistics.mean_abs_residual) &&
        std::isfinite(statistics.match_ratio) &&
        statistics.match_ratio >= 0.0 && statistics.match_ratio <= 1.0;
  }
  return statistics;
}

VoxelMapManager::RepresentativeMapStatistics
VoxelMapManager::ExportRepresentativeCloud(
    PointCloudXYZI &cloud, const std::size_t maximum_points,
    const std::size_t samples_per_plane) const
{
  if (maximum_points == 0)
    throw std::invalid_argument("representative map maximum_points must be positive");
  if (samples_per_plane == 0 || samples_per_plane > 9)
    throw std::invalid_argument(
        "representative map samples_per_plane must be in [1,9]");

  RepresentativeMapStatistics statistics;
  statistics.root_voxels = voxel_map_.size();
  statistics.maximum_points = maximum_points;
  cloud.clear();

  std::vector<std::pair<VOXEL_LOCATION, const VoxelOctoTree *>> roots;
  roots.reserve(voxel_map_.size());
  for (const auto &entry : voxel_map_) roots.emplace_back(entry.first, entry.second);
  std::sort(roots.begin(), roots.end(),
      [](const auto &left, const auto &right) {
        if (left.first.x != right.first.x) return left.first.x < right.first.x;
        if (left.first.y != right.first.y) return left.first.y < right.first.y;
        return left.first.z < right.first.z;
      });
  for (const auto &entry : roots)
    countRepresentativeNode(entry.second, statistics);

  std::size_t effective_samples_per_plane = 1;
  if (statistics.plane_nodes > 0 &&
      statistics.terminal_nodes < maximum_points)
  {
    effective_samples_per_plane = std::min(
        samples_per_plane,
        1 + (maximum_points - statistics.terminal_nodes) /
                statistics.plane_nodes);
  }
  const std::size_t expected_points = statistics.fallback_nodes +
      statistics.plane_nodes * effective_samples_per_plane;
  cloud.reserve(std::min(maximum_points, expected_points));
  for (const auto &entry : roots)
  {
    exportRepresentativeNode(entry.second, maximum_points,
                             effective_samples_per_plane, cloud, statistics);
    if (statistics.truncated) break;
  }
  cloud.width = static_cast<std::uint32_t>(cloud.size());
  cloud.height = 1;
  cloud.is_dense = true;
  statistics.emitted_points = cloud.size();
  return statistics;
}

void VoxelMapManager::build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess,
                                            double &prob, PointToPlane &single_ptpl) const
{
  int max_layer = config_setting_.max_layer_;
  double sigma_num = config_setting_.sigma_num_;

  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_w;
  if (current_octo->plane_ptr_->is_plane_)
  {
    VoxelPlane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center_;
    float dis_to_plane = fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_);
    float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) + (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                          (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius_)
    {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
      J_nq.block<1, 3>(0, 3) = -plane.normal_;
      double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
      sigma_l += plane.normal_.transpose() * pv.var * plane.normal_;
      if (dis_to_plane < sigma_num * sqrt(sigma_l))
      {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob)
        {
          prob = this_prob;
          pv.normal = plane.normal_;
          single_ptpl.body_cov_ = pv.body_var;
          single_ptpl.point_b_ = pv.point_b;
          single_ptpl.point_w_ = pv.point_w;
          single_ptpl.plane_var_ = plane.plane_var_;
          single_ptpl.normal_ = plane.normal_;
          single_ptpl.center_ = plane.center_;
          single_ptpl.d_ = plane.d_;
          single_ptpl.layer_ = current_layer;
          single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        }
        return;
      }
      else
      {
        // is_sucess = false;
        return;
      }
    }
    else
    {
      // is_sucess = false;
      return;
    }
  }
  else
  {
    if (current_layer < max_layer)
    {
      for (size_t leafnum = 0; leafnum < 8; leafnum++)
      {
        if (current_octo->leaves_[leafnum] != nullptr)
        {

          VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl);
        }
      }
      return;
    }
    else { return; }
  }
}

void VoxelMapManager::pubVoxelMap()
{
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<VoxelPlane> pub_plane_list;
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++)
  {
    GetUpdatePlane(iter->second, config_setting_.max_layer_, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++)
  {
    V3D plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) { trace = max_trace; }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane_) { alpha = use_alpha; }
    else { alpha = 0; }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  voxel_map_pub_.publish(voxel_plane);
  loop.sleep();
}

void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list)
{
  if (current_octo->layer_ > pub_max_voxel_layer) { return; }
  if (current_octo->plane_ptr_->is_update_) { plane_list.push_back(*current_octo->plane_ptr_); }
  if (current_octo->layer_ < current_octo->max_layer_)
  {
    if (!current_octo->plane_ptr_->is_plane_)
    {
      for (size_t i = 0; i < 8; i++)
      {
        if (current_octo->leaves_[i] != nullptr) { GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list); }
      }
    }
  }
  return;
}

void VoxelMapManager::pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane,
                                     const float alpha, const Eigen::Vector3d rgb)
{
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id_;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center_[0];
  plane.pose.position.y = single_plane.center_[1];
  plane.pose.position.z = single_plane.center_[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal_, single_plane.y_normal_, single_plane.normal_, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value_);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value_);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value_);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void VoxelMapManager::CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                                      geometry_msgs::Quaternion &q)
{
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) { v = vmin; }

  if (v > vmax) { v = vmax; }

  double dr, dg, db;

  if (v < 0.1242)
  {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  }
  else if (v < 0.3747)
  {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  }
  else if (v < 0.6253)
  {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  }
  else if (v < 0.8758)
  {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  }
  else
  {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

void VoxelMapManager::mapSliding()
{
  if((position_last_ - last_slide_position).norm() < config_setting_.sliding_thresh)
  {
    std::cout<<RED<<"[DEBUG]: Last sliding length "<<(position_last_ - last_slide_position).norm()<<RESET<<"\n";
    return;
  }

  //get global id now
  last_slide_position = position_last_;
  double t_sliding_start = omp_get_wtime();
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = position_last_[j] / config_setting_.max_voxel_size_;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  // VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);//discrete global
  clearMemOutOfMap((int64_t)loc_xyz[0] + config_setting_.half_map_size, (int64_t)loc_xyz[0] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[1] + config_setting_.half_map_size, (int64_t)loc_xyz[1] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[2] + config_setting_.half_map_size, (int64_t)loc_xyz[2] - config_setting_.half_map_size);
  double t_sliding_end = omp_get_wtime();
  std::cout<<RED<<"[DEBUG]: Map sliding using "<<t_sliding_end - t_sliding_start<<" secs"<<RESET<<"\n";
  return;
}

void VoxelMapManager::clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min )
{
  int delete_voxel_cout = 0;
  // double delete_time = 0;
  // double last_delete_time = 0;
  for (auto it = voxel_map_.begin(); it != voxel_map_.end(); )
  {
    const VOXEL_LOCATION& loc = it->first;
    bool should_remove = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min || loc.z > z_max || loc.z < z_min;
    if (should_remove){
      RecordMutation(loc);
      // last_delete_time = omp_get_wtime();
      delete it->second;
      it = voxel_map_.erase(it);
      // delete_time += omp_get_wtime() - last_delete_time;
      delete_voxel_cout++;
    } else {
      ++it;
    }
  }
  std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" root voxels"<<RESET<<"\n";
  // std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" voxels using "<<delete_time<<" s"<<RESET<<"\n";
}
