// Included in LIVMapper.cpp so the diagnostic uses the production component
// checkpoint codecs and the exact normal Process2/LIO/VIO update path.
#include <ros/serialization.h>

namespace nominal_probe {
using Bytes = std::vector<std::uint8_t>;
void writeEvent(const std::string &path, const LidarMeasureGroup &m) {
  Writer w;
  w.pod(m.lidar_frame_beg_time); w.pod(m.lidar_frame_end_time);
  w.pod(m.last_lio_update_time); w.pod(static_cast<int>(m.lio_vio_flg));
  w.pod(m.lidar_scan_index_now);
  writeMapperCloud(w,*m.lidar); writeMapperCloud(w,*m.pcl_proc_cur); writeMapperCloud(w,*m.pcl_proc_next);
  w.pod(static_cast<std::uint64_t>(m.measures.size()));
  for(const auto &g:m.measures) {
    w.pod(g.lio_time); w.pod(g.vio_time);
    w.pod(g.img.rows); w.pod(g.img.cols); w.pod(g.img.type());
    for(int r=0;r<g.img.rows;++r) w.raw(g.img.ptr(r),g.img.cols*g.img.elemSize());
    w.pod(static_cast<std::uint64_t>(g.imu.size()));
    for(const auto &imu:g.imu) {
      Bytes message(ros::serialization::serializationLength(*imu));
      ros::serialization::OStream s(message.data(),message.size());
      ros::serialization::serialize(s,*imu); writeBlob(w,message);
    }
  }
  Writer envelope;
  envelope.pod(std::uint64_t(0x31455645504d4f4eULL));
  envelope.pod(fast_livo::checkpoint_io::fnv1a(w.bytes()));
  envelope.raw(w.bytes().data(),w.bytes().size());
  std::ofstream f(path,std::ios::binary|std::ios::trunc);
  f.write(reinterpret_cast<const char*>(envelope.bytes().data()),envelope.bytes().size());
  f.flush(); if(!f) throw std::runtime_error("nominal event archive write failed");
}
LidarMeasureGroup readEvent(const std::string &path) {
  std::ifstream f(path,std::ios::binary);
  if(!f) throw std::runtime_error("nominal event archive absent");
  Bytes bytes((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
  Reader envelope(bytes);
  if(envelope.pod<std::uint64_t>()!=0x31455645504d4f4eULL) throw std::runtime_error("bad nominal event magic");
  const auto checksum=envelope.pod<std::uint64_t>();
  Bytes payload(bytes.begin()+envelope.cursor(),bytes.end());
  if(fast_livo::checkpoint_io::fnv1a(payload)!=checksum) throw std::runtime_error("nominal event checksum mismatch");
  Reader r(payload); LidarMeasureGroup m;
  m.lidar_frame_beg_time=r.pod<double>(); m.lidar_frame_end_time=r.pod<double>();
  m.last_lio_update_time=r.pod<double>(); m.lio_vio_flg=static_cast<EKF_STATE>(r.pod<int>());
  m.lidar_scan_index_now=r.pod<int>();
  // Live synchronization scratch appends points directly and may retain stale
  // width/height. Preserve these descriptors exactly; rectangular-cloud checks
  // remain mandatory for the production mapper checkpoint decoder.
  *m.lidar=readMapperCloud(r,false); *m.pcl_proc_cur=readMapperCloud(r,false); *m.pcl_proc_next=readMapperCloud(r,false);
  const auto groups=fast_livo::checkpoint_io::checkedCount(r,1000,"nominal measure group");
  for(std::uint64_t k=0;k<groups;++k) {
    MeasureGroup g; g.lio_time=r.pod<double>(); g.vio_time=r.pod<double>();
    const int rows=r.pod<int>(), cols=r.pod<int>(), type=r.pod<int>();
    if(rows<0 || rows>20000 || cols<0 || cols>20000) throw std::runtime_error("invalid nominal image dimensions");
    if(rows && cols) { g.img.create(rows,cols,type); for(int y=0;y<rows;++y) r.raw(g.img.ptr(y),cols*g.img.elemSize()); }
    const auto count=fast_livo::checkpoint_io::checkedCount(r,100000,"nominal IMU");
    for(std::uint64_t i=0;i<count;++i) {
      const auto n=fast_livo::checkpoint_io::checkedCount(r,1000000,"nominal IMU bytes");
      Bytes message(n); r.raw(message.data(),message.size());
      sensor_msgs::Imu::Ptr imu(new sensor_msgs::Imu());
      ros::serialization::IStream s(message.data(),message.size()); ros::serialization::deserialize(s,*imu);
      g.imu.push_back(imu);
    }
    m.measures.push_back(std::move(g));
  }
  if(!r.finished()) throw std::runtime_error("trailing nominal event bytes");
  return m;
}
std::string eventPath(const std::string &root,std::uint64_t id) {
  return joinPath(root,"event_"+std::to_string(id)+".bin");
}
}

LIVMapper::RecoveryHandoffBundle LIVMapper::captureNominalProbeBundle() {
  RecoveryHandoffBundle b;
  b.generation=1; b.boundary_relative_time=last_completed_event_relative_time;
  b.next_event_sequence=fusion_audit_event; b.visual_patch_size_total=vio_manager->patch_size_total;
  b.mapper=captureMapperCheckpoint(); b.voxel_map=voxelmap_manager->CreateSnapshot();
  b.visual_map=vio_manager->createCheckpoint(); b.detector=harm_detector->createSnapshot();
  return b;
}
LidarMeasureGroup LIVMapper::cloneNominalProbeEvent(const LidarMeasureGroup &s) const {
  auto m=cloneMeasureGroup(s);
  // Preserve even live synchronization scratch for exact input diagnostics.
  *m.lidar=*s.lidar; *m.pcl_proc_next=*s.pcl_proc_next;
  return m;
}
void LIVMapper::nominalProbeBeforeEvent(double t) {
  if(nominal_probe_directory.empty()) return;
  if(!diagnostic_forbid_recovery || map_transaction_config.enabled ||
      !fusion_audit.enabled() || nominal_probe_end<=nominal_probe_start)
    throw std::runtime_error("nominal probe requires audited baseline, no transactions and forbid_recovery");
  if(!nominal_probe_seed && lidar_map_inited && LidarMeasures.lio_vio_flg==LIO &&
      last_completed_event_relative_time>=nominal_probe_start) {
    if(::mkdir(nominal_probe_directory.c_str(),0755)!=0 && errno!=EEXIST)
      throw std::runtime_error("cannot create nominal probe directory");
    nominal_probe_seed.reset(new RecoveryHandoffBundle(captureNominalProbeBundle()));
    writeRecoveryHandoffBundleAtomic(joinPath(nominal_probe_directory,"seed.bin"),*nominal_probe_seed);
  }
  if(!nominal_probe_seed) return;
  auto event=cloneNominalProbeEvent(LidarMeasures);
  nominal_probe::writeEvent(nominal_probe::eventPath(nominal_probe_directory,fusion_audit_event),event);
  nominal_probe_events.emplace_back(fusion_audit_event,std::move(event));
}
bool LIVMapper::nominalProbeAfterEvent(double t) {
  if(!nominal_probe_seed || t<nominal_probe_end || LidarMeasures.lio_vio_flg!=VIO) return false;
  auto direct=captureNominalProbeBundle();
  writeRecoveryHandoffBundleAtomic(joinPath(nominal_probe_directory,"terminal_direct.bin"),direct);
  std::ofstream summary(joinPath(nominal_probe_directory,"map_fingerprints.csv"));
  summary << "route,voxel_hash,visual_hash\n";
  auto fingerprint=[&](const std::string &route,RecoveryHandoffBundle &b) {
    // Profiling clocks are not estimator state and are intentionally excluded.
    b.visual_map->average_total_time=0.0;
    summary << route << ',' << b.voxel_map->fingerprint() << ','
      << fast_livo::checkpoint_io::fnv1a(vio_manager->serializeCheckpoint(*b.visual_map)) << '\n';
    summary.flush();
  };
  fingerprint("direct",direct);
  const auto terminal_id=fusion_audit_event;
  for(int serialized=0;serialized<2;++serialized) for(int repeat=1;repeat<=3;++repeat) {
    const std::string route=std::string(serialized?"serialized_":"memory_")+std::to_string(repeat);
    std::unique_ptr<RecoveryHandoffBundle> decoded;
    if(serialized) decoded.reset(new RecoveryHandoffBundle(readRecoveryHandoffBundle(joinPath(nominal_probe_directory,"seed.bin"))));
    const auto &seed=serialized?*decoded:*nominal_probe_seed;
    voxelmap_manager->RestoreSnapshot(*seed.voxel_map);
    restoreMapperCheckpoint(seed.mapper);
    vio_manager->restoreCheckpoint(*seed.visual_map);
    harm_detector->restoreSnapshot(seed.detector);
    for(const auto &item:nominal_probe_events) {
      fusion_audit_event=item.first;
      LidarMeasures=serialized?nominal_probe::readEvent(nominal_probe::eventPath(nominal_probe_directory,item.first)):
                              cloneNominalProbeEvent(item.second);
      fusion_audit.write(item.first,route,"synced",LidarMeasures,_first_lidar_time,_state,feats_undistort);
      fusion_audit.write(item.first,route,"pre_imu",LidarMeasures,_first_lidar_time,_state,feats_undistort);
      processImu();
      fusion_audit.write(item.first,route,"post_imu",LidarMeasures,_first_lidar_time,_state,feats_undistort);
      stateEstimationAndMapping();
      fusion_audit.write(item.first,route,"post_update",LidarMeasures,_first_lidar_time,_state,feats_undistort);
    }
    auto terminal=captureNominalProbeBundle();
    writeRecoveryHandoffBundleAtomic(joinPath(nominal_probe_directory,"terminal_"+route+".bin"),terminal);
    fingerprint(route,terminal);
  }
  fusion_audit_event=terminal_id;
  std::ofstream done(joinPath(nominal_probe_directory,"execution_complete.json"));
  done << "{\"executed_routes\":6,\"events_per_route\":" << nominal_probe_events.size()
       << ",\"equivalence_passed\":null,\"note\":\"execution only; compare to direct before acceptance\"}\n";
  done.flush(); if(!done) throw std::runtime_error("nominal probe completion write failed");
  return true; // Diagnostic process ends. No recovered state is made public.
}
