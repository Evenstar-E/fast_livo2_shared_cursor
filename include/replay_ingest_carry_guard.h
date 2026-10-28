#pragma once
#include <utility>

namespace fast_livo { namespace replay {
// A replay event intentionally omits live synchronization scratch. Keep that
// ownership outside event materialization and restore it on commit, rejection,
// exception, or early return. Do NOT roll back the processed event's timestamp,
// LIO/VIO phase, current cloud, or estimator state when reconnecting to ingestion.
template<class MeasureGroup> class IngestCarryGuard {
  MeasureGroup &group_;
  decltype(std::declval<MeasureGroup>().pcl_proc_next) next_;
  decltype(std::declval<MeasureGroup>().lidar) lidar_;
  decltype(std::declval<MeasureGroup>().lidar_scan_index_now) scan_index_;
public:
  explicit IngestCarryGuard(MeasureGroup &group)
      : group_(group),next_(group.pcl_proc_next),lidar_(group.lidar),
        scan_index_(group.lidar_scan_index_now) {}
  ~IngestCarryGuard() noexcept {
    group_.pcl_proc_next=std::move(next_);
    group_.lidar=std::move(lidar_);
    group_.lidar_scan_index_now=scan_index_;
  }
  IngestCarryGuard(const IngestCarryGuard&)=delete;
  IngestCarryGuard& operator=(const IngestCarryGuard&)=delete;
};
}}
