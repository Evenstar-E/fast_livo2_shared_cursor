// Diagnostic-only oracle input attribution. Included after nominal event codec.
namespace causal_probe {
std::string key(const LidarMeasureGroup &m) {
  if(m.measures.empty()) throw std::runtime_error("causal event lacks measurements");
  const auto &g=m.measures.back();
  const double t=m.lio_vio_flg==VIO?g.vio_time:g.lio_time;
  return std::to_string(static_cast<long long>(std::llround(t*1e9)))+"_"+
         std::to_string(static_cast<int>(m.lio_vio_flg));
}
void sameNonCloudInput(const LidarMeasureGroup &a,const LidarMeasureGroup &b) {
  if(a.lio_vio_flg!=b.lio_vio_flg || a.last_lio_update_time!=b.last_lio_update_time ||
     a.lidar_frame_beg_time!=b.lidar_frame_beg_time || a.lidar_frame_end_time!=b.lidar_frame_end_time ||
     a.measures.size()!=b.measures.size()) throw std::runtime_error("causal reference timing differs");
  for(std::size_t k=0;k<a.measures.size();++k) {
    const auto &u=a.measures[k];const auto &v=b.measures[k];
    if(u.lio_time!=v.lio_time || u.vio_time!=v.vio_time || u.imu.size()!=v.imu.size() ||
       u.img.size()!=v.img.size() || u.img.type()!=v.img.type())
      throw std::runtime_error("causal reference event descriptors differ");
    if(!u.img.empty() && cv::norm(u.img,v.img,cv::NORM_INF)!=0)
      throw std::runtime_error("causal reference image differs");
    for(std::size_t j=0;j<u.imu.size();++j) {
      std::vector<std::uint8_t> ub(ros::serialization::serializationLength(*u.imu[j]));
      std::vector<std::uint8_t> vb(ros::serialization::serializationLength(*v.imu[j]));
      ros::serialization::OStream us(ub.data(),ub.size()),vs(vb.data(),vb.size());
      ros::serialization::serialize(us,*u.imu[j]);ros::serialization::serialize(vs,*v.imu[j]);
      if(ub!=vb) throw std::runtime_error("causal reference IMU differs");
    }
  }
}
}

void LIVMapper::causalArchiveDirectEvent(double t) {
  if(causal_event_archive_directory.empty() || t<causal_archive_start || t>causal_archive_end) return;
  if(!diagnostic_forbid_recovery || map_transaction_config.enabled)
    throw std::runtime_error("clean causal archive requires baseline with recovery forbidden");
  nominal_probe::writeEvent(joinPath(causal_event_archive_directory,causal_probe::key(LidarMeasures)+".bin"),LidarMeasures);
}

void LIVMapper::causalReplaceReplayCloud(double t) {
  if(causal_reference_event_directory.empty() || LidarMeasures.lio_vio_flg!=LIO) return;
  if(!replay_range_scale_oracle_config.enabled || healthy_tail_config.enabled || recovery_benefit_config.enabled)
    throw std::runtime_error("exact-cloud substitution is only allowed in isolated oracle diagnostics");
  if(t<causal_archive_start || t>causal_archive_end)
    throw std::runtime_error("causal reference window does not cover replay");
  auto reference=nominal_probe::readEvent(joinPath(causal_reference_event_directory,causal_probe::key(LidarMeasures)+".bin"));
  causal_probe::sameNonCloudInput(LidarMeasures,reference);
  // Replace only consumed LiDAR geometry. Never overwrite IMUs/images/state,
  // timing, checkpoint/map contents, or the live ingest carry-over queues.
  LidarMeasures.pcl_proc_cur.reset(new PointCloudXYZI(*reference.pcl_proc_cur));
  if(!imu_en) LidarMeasures.lidar.reset(new PointCloudXYZI(*reference.lidar));
}

void LIVMapper::causalDumpCloud(const std::string &route,const std::string &stage,
                              double t,const PointCloudXYZI::Ptr &cloud) const {
  if(causal_cloud_audit_directory.empty() || t<79.8 || t>80.5 || LidarMeasures.lio_vio_flg!=LIO) return;
  Writer w;w.pod(static_cast<std::uint64_t>(cloud?cloud->size():0));
  if(cloud) for(const auto &p:cloud->points) {
    w.pod(p.x);w.pod(p.y);w.pod(p.z);w.pod(p.intensity);w.pod(p.curvature);
    w.pod(p.normal_x);w.pod(p.normal_y);w.pod(p.normal_z);
  }
  const auto file=joinPath(causal_cloud_audit_directory,causal_probe::key(LidarMeasures)+"_"+route+"_"+stage+".bin");
  std::ofstream f(file,std::ios::binary|std::ios::trunc);
  f.write(reinterpret_cast<const char*>(w.bytes().data()),w.bytes().size());
  f.flush();if(!f) throw std::runtime_error("causal cloud audit write failed");
}
