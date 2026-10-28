#include "LIVMapper.h"

#include <cstdlib>
#include <cstdio>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh); 
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();
  // run() has already saved persistent products and the append-only audit
  // journals flush every row.  Returning from main would synchronously destroy
  // millions of octree allocations and can exceed roslaunch's shutdown budget
  // even though all scientific work is complete.  A process-isolated mapper
  // has no in-process successor that can observe those destructors; exit with
  // success and let the kernel reclaim its address space in bulk.
  std::fflush(nullptr);
  // std::exit still invokes atexit handlers and static-duration destructors.
  // In the process-isolated shadow lifecycle those ROS/PCL cleanup paths can
  // race after savePCD() has completed, which was observed as a post-save
  // SIGSEGV in both mapper generations.  _Exit performs the kernel-reclaim
  // fast exit promised above and cannot change any already-flushed artifact.
  std::_Exit(EXIT_SUCCESS);
}
