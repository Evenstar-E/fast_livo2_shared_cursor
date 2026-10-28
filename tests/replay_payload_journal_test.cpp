#include "replay_payload_journal.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <unistd.h>

int main()
{
  const std::string path = "/tmp/fast_livo_replay_payload_test_" +
      std::to_string(static_cast<long long>(::getpid())) + ".bin";
  fast_livo::replay_storage::Journal journal(path);
  journal.reset();

  LidarMeasureGroup group;
  group.lidar_frame_beg_time = 1.25;
  group.lidar_frame_end_time = 1.35;
  group.last_lio_update_time = 1.34;
  group.lio_vio_flg = LIO;
  group.lidar_scan_index_now = 7;
  group.raw_reprocess_begin_time = 1.25;
  group.raw_reprocess_end_time = 1.34;
  group.raw_reprocess_recipe_valid = true;
  group.raw_reprocess_split_by_time = true;
  group.raw_preprocess_lidar_type = XT32;
  group.raw_preprocess_point_filter_num = 1;
  group.raw_preprocess_scan_lines = 32;
  group.raw_preprocess_blind = 0.5;
  group.raw_preprocess_feature_enabled = false;

  RawLidarFrame raw;
  raw.message_type = RawLidarMessageType::POINTCLOUD2;
  raw.sensor_header_time = 1.2;
  raw.effective_header_time = 1.2;
  raw.minimum_processed_offset_ms = 0.0f;
  raw.maximum_processed_offset_ms = 100.0f;
  raw.oracle_source_relative_time = 80.1;
  raw.oracle_inverse_range_scale = 0.98;
  raw.oracle_range_scale_active = true;
  raw.serialized_checksum = 1234;
  raw.serialized_message = {1, 2, 3, 4, 5};
  group.raw_lidar_frames.push_back(raw);

  PointType point;
  point.x = 1.0f;
  point.y = -2.0f;
  point.z = 3.5f;
  point.intensity = 42.0f;
  point.normal_x = 0.1f;
  point.normal_y = 0.2f;
  point.normal_z = 0.3f;
  point.curvature = 9.0f;
  point.data_c[2] = 123.0f; // Non-semantic PCL storage must not reach disk.
  point.data_c[3] = 456.0f;
  group.pcl_proc_cur->points.push_back(point);
  group.pcl_proc_cur->width = 1;
  group.pcl_proc_cur->height = 1;
  group.pcl_proc_cur->is_dense = true;

  MeasureGroup measure;
  measure.vio_time = 1.3;
  measure.lio_time = 1.34;
  measure.img = cv::Mat(2, 3, CV_8UC1);
  for (int i = 0; i < 6; ++i) measure.img.data[i] = static_cast<unsigned char>(i + 10);
  sensor_msgs::Imu::Ptr imu(new sensor_msgs::Imu());
  imu->header.stamp = ros::Time(1.3);
  imu->linear_acceleration.x = 4.0;
  measure.imu.push_back(imu);
  group.measures.push_back(measure);
  const LidarMeasureGroup original = group;

  const auto record = journal.spill(group);
  assert(record.spilled);
  assert(record.size > 0);
  assert(group.pcl_proc_cur->empty());
  assert(group.raw_lidar_frames.empty());
  assert(group.measures.front().img.empty());
  assert(journal.recordsWritten() == 1);

  const LidarMeasureGroup restored = journal.materialize(group, record);
  assert(restored.lidar_frame_beg_time == 1.25);
  assert(restored.lio_vio_flg == LIO);
  assert(restored.pcl_proc_cur->points.size() == 1);
  const PointType &restored_point = restored.pcl_proc_cur->points.front();
  assert(restored_point.x == point.x && restored_point.y == point.y &&
         restored_point.z == point.z && restored_point.data[3] == point.data[3]);
  assert(restored_point.normal_x == point.normal_x &&
         restored_point.normal_y == point.normal_y &&
         restored_point.normal_z == point.normal_z &&
         restored_point.data_n[3] == point.data_n[3]);
  assert(restored_point.intensity == point.intensity &&
         restored_point.curvature == point.curvature);
  assert(restored_point.data_c[2] == 0.0f &&
         restored_point.data_c[3] == 0.0f);
  assert(restored.pcl_proc_cur->width == 1);
  assert(restored.pcl_proc_cur->height == 1);
  assert(restored.pcl_proc_cur->is_dense);
  assert(restored.raw_reprocess_recipe_valid);
  assert(restored.raw_reprocess_split_by_time);
  assert(restored.raw_preprocess_lidar_type == XT32);
  assert(restored.raw_lidar_frames.size() == 1);
  assert(restored.raw_lidar_frames.front().serialized_message ==
         raw.serialized_message);
  assert(restored.raw_lidar_frames.front().serialized_checksum == 1234);
  assert(restored.measures.size() == 1);
  assert(restored.measures.front().img.rows == 2);
  assert(restored.measures.front().img.cols == 3);
  for (int i = 0; i < 6; ++i)
    assert(restored.measures.front().img.data[i] == static_cast<unsigned char>(i + 10));
  assert(restored.measures.front().imu.front()->linear_acceleration.x == 4.0);
  assert(journal.recordsRead() == 1);

  const std::string async_path = path + ".async";
  fast_livo::replay_storage::Journal async_journal(async_path);
  async_journal.configureAsync(true, 1024 * 1024);
  async_journal.reset();
  LidarMeasureGroup async_group = original;
  async_group.pcl_proc_cur.reset(new PointCloudXYZI(*original.pcl_proc_cur));
  async_group.pcl_proc_cur->points.front().data_c[2] = -321.0f;
  async_group.pcl_proc_cur->points.front().data_c[3] = -654.0f;
  const auto async_record = async_journal.spill(async_group);
  // Reading immediately after enqueue must block until the exact record is
  // visible, without requiring a full durable commit.
  const LidarMeasureGroup async_restored =
      async_journal.materialize(async_group, async_record);
  assert(async_restored.raw_lidar_frames.front().serialized_message ==
         raw.serialized_message);
  assert(async_restored.pcl_proc_cur->points.size() == 1);
  assert(async_restored.pcl_proc_cur->points.front().x == point.x);
  assert(async_restored.pcl_proc_cur->points.front().intensity == point.intensity);
  assert(async_restored.pcl_proc_cur->points.front().data_c[2] == 0.0f);
  assert(async_restored.pcl_proc_cur->points.front().data_c[3] == 0.0f);
  assert(async_journal.durableBytes() == 0);
  async_journal.commitDurable();
  assert(async_journal.durableBytes() == async_record.size);
  assert(async_journal.peakQueuedBytes() <= 1024 * 1024);

  journal.commitDurable();
  std::ifstream sync_input(path, std::ios::binary);
  std::ifstream async_input(async_path, std::ios::binary);
  const std::vector<char> sync_bytes((std::istreambuf_iterator<char>(sync_input)),
                                     std::istreambuf_iterator<char>());
  const std::vector<char> async_bytes((std::istreambuf_iterator<char>(async_input)),
                                      std::istreambuf_iterator<char>());
  assert(sync_bytes == async_bytes);

  std::uint64_t previous_end = async_record.offset + async_record.size;
  for (int index = 0; index < 32; ++index)
  {
    LidarMeasureGroup next = original;
    next.raw_lidar_frames.front().serialized_message.push_back(
        static_cast<unsigned char>(index));
    const auto record_next = async_journal.spill(next);
    LidarMeasureGroup sync_next = original;
    sync_next.raw_lidar_frames.front().serialized_message.push_back(
        static_cast<unsigned char>(index));
    const auto sync_record_next = journal.spill(sync_next);
    assert(sync_record_next.offset == record_next.offset);
    assert(sync_record_next.size == record_next.size);
    assert(sync_record_next.checksum == record_next.checksum);
    assert(record_next.offset == previous_end);
    previous_end += record_next.size;
    const auto restored_next = async_journal.materialize(next, record_next);
    assert(restored_next.raw_lidar_frames.front().serialized_message.back() ==
           static_cast<unsigned char>(index));
  }
  async_journal.commitDurable();
  journal.commitDurable();
  assert(async_journal.durableBytes() == previous_end);
  assert(async_journal.peakQueuedBytes() <= 1024 * 1024);
  std::ifstream sync_all(path, std::ios::binary);
  std::ifstream async_all(async_path, std::ios::binary);
  assert(std::vector<char>((std::istreambuf_iterator<char>(sync_all)),
                           std::istreambuf_iterator<char>()) ==
         std::vector<char>((std::istreambuf_iterator<char>(async_all)),
                           std::istreambuf_iterator<char>()));

  const std::string bounded_path = path + ".bounded";
  fast_livo::replay_storage::Journal bounded_journal(bounded_path);
  const std::uint64_t queue_limit = async_record.size * 2;
  bounded_journal.configureAsync(true, queue_limit);
  bounded_journal.reset();
  for (int index = 0; index < 128; ++index)
  {
    LidarMeasureGroup queued = original;
    bounded_journal.spill(queued);
    assert(bounded_journal.peakQueuedBytes() <= queue_limit);
  }
  // Reset must drain an in-flight bounded queue before truncating the file.
  bounded_journal.reset();
  assert(bounded_journal.recordsWritten() == 0);
  std::ifstream bounded_after_reset(bounded_path, std::ios::binary | std::ios::ate);
  assert(bounded_after_reset.tellg() == 0);

  const std::string tiny_path = path + ".tiny";
  fast_livo::replay_storage::Journal tiny_journal(tiny_path);
  tiny_journal.configureAsync(true, 1);
  tiny_journal.reset();
  LidarMeasureGroup too_large = original;
  bool oversize_failed = false;
  try { tiny_journal.spill(too_large); }
  catch (const std::runtime_error &) { oversize_failed = true; }
  assert(oversize_failed);
  assert(!too_large.raw_lidar_frames.empty());
  tiny_journal.reset();

  journal.reset();
  assert(journal.bytesWritten() == 0);
  assert(journal.recordsWritten() == 0);
  assert(journal.recordsRead() == 0);
  std::remove(path.c_str());
  std::remove(async_path.c_str());
  std::remove(tiny_path.c_str());
  std::remove(bounded_path.c_str());
  return 0;
}
