#include "LIVMapper.h"

#include <pcl/io/pcd_io.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace {

bool exists(const std::string &path)
{
  struct stat info{};
  return ::stat(path.c_str(), &info) == 0;
}

void writeVector(std::ostream &out, const V3D &value)
{
  out << '[' << value.x() << ',' << value.y() << ',' << value.z() << ']';
}

}  // namespace

int main(int argc, char **argv)
{
  if (argc != 4)
  {
    std::cerr << "usage: export_recovery_handoff_snapshot "
              << "HANDOFF.bin OUTPUT.pcd OUTPUT.json\n";
    return 2;
  }
  const std::string handoff_path = argv[1];
  const std::string cloud_path = argv[2];
  const std::string receipt_path = argv[3];
  if (!exists(handoff_path) || exists(cloud_path) || exists(receipt_path))
  {
    std::cerr << "missing input or output already exists\n";
    return 2;
  }
  try
  {
    LIVMapper::RecoveryHandoffBundle handoff =
        LIVMapper::readRecoveryHandoffBundle(handoff_path);
    if (!handoff.voxel_map)
      throw std::runtime_error("handoff has no voxel snapshot");
    VoxelMapConfig config{};
    std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> empty_map;
    VoxelMapManager manager(config, empty_map);
    manager.ConsumeSnapshot(*handoff.voxel_map);
    PointCloudXYZI cloud;
    constexpr std::size_t kMaximumPoints = 5000000;
    const auto stats = manager.ExportRepresentativeCloud(
        cloud, kMaximumPoints, 1);
    if (stats.truncated || cloud.empty())
      throw std::runtime_error("checkpoint map is empty or truncated");
    if (pcl::io::savePCDFileBinary(cloud_path, cloud) != 0)
      throw std::runtime_error("cannot save checkpoint representative map");
    std::ofstream out(receipt_path.c_str(), std::ios::out | std::ios::trunc);
    if (!out)
      throw std::runtime_error("cannot save checkpoint receipt");
    out << std::setprecision(17);
    out << "{\n  \"schema\": 1,\n"
        << "  \"generation\": " << handoff.generation << ",\n"
        << "  \"boundary_label_relative_time_sec\": "
        << handoff.boundary_relative_time << ",\n"
        << "  \"completed_event_relative_time_sec\": "
        << std::nextafter(handoff.boundary_relative_time, INFINITY) << ",\n"
        << "  \"voxel_root_count\": " << stats.root_voxels << ",\n"
        << "  \"terminal_node_count\": " << stats.terminal_nodes << ",\n"
        << "  \"plane_node_count\": " << stats.plane_nodes << ",\n"
        << "  \"representative_point_count\": " << stats.emitted_points
        << ",\n  \"representative_truncated\": false,\n"
        << "  \"state\": {\n    \"position\": ";
    const StatesGroup &state = handoff.mapper.state;
    writeVector(out, state.pos_end);
    out << ",\n    \"velocity\": ";
    writeVector(out, state.vel_end);
    out << ",\n    \"gyro_bias\": ";
    writeVector(out, state.bias_g);
    out << ",\n    \"accel_bias\": ";
    writeVector(out, state.bias_a);
    out << ",\n    \"gravity\": ";
    writeVector(out, state.gravity);
    out << ",\n    \"rotation_matrix_row_major\": [";
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col)
      {
        if (row || col) out << ',';
        out << state.rot_end(row, col);
      }
    out << "]\n  }\n}\n";
    out.flush();
    if (!out) throw std::runtime_error("checkpoint receipt write failed");
    manager.ClearVoxelMap();
    std::cout << "EXPORTED generation=" << handoff.generation
              << " roots=" << stats.root_voxels
              << " points=" << stats.emitted_points << '\n';
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "EXPORT_FAILED " << error.what() << '\n';
    return 1;
  }
}
