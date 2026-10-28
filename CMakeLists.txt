cmake_minimum_required(VERSION 2.8.3)
project(fast_livo)

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set common compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")

# Specific settings for Debug build
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Detect CPU architecture
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# Specific settings for Release build
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # 64-bit ARM optimizations (e.g., RK3588 and Jetson Orin NX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
    message(STATUS "Using 64-bit ARM optimizations: -O3 -mcpu=native -mtune=native -ffast-math")
  else()
    # 32-bit ARM optimizations with NEON support
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
    message(STATUS "Using 32-bit ARM optimizations: -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
  # x86-64 (Intel/AMD) optimizations
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops") #-flto
  message(STATUS "Using general x86 optimizations: -O3 -march=native -mtune=native -funroll-loops") 
  add_definitions(-DX86_ARCH)
endif()

# Define project root directory
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# Detect CPU core count for potential multithreading optimization
include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor count: ${N}")

# Set the number of cores for multithreading
if(N GREATER 4)
  math(EXPR PROC_NUM "4")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
elseif(N GREATER 1)
  math(EXPR PROC_NUM "${N}")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
  message(STATUS "Multithreading enabled. Cores: ${PROC_NUM}")
else()
  add_definitions(-DMP_PROC_NUM=1)
  message(STATUS "Single core detected. Multithreading disabled.")
endif()

# Check for OpenMP support
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  add_compile_options(${OpenMP_CXX_FLAGS})
else()
  message(STATUS "OpenMP not found, proceeding without it")
endif()

# Check for mimalloc support
find_package(mimalloc QUIET)
if(mimalloc_FOUND)
  message(STATUS "mimalloc found")
else()
  message(STATUS "mimalloc not found, proceeding without it")
endif()

# Find catkin and required dependencies
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenSSL REQUIRED)

set(Sophus_LIBRARIES libSophus.so)

# Define the catkin package
catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge vikit_common vikit_ros image_transport
  DEPENDS EIGEN3 PCL OpenCV Sophus
)

# Include directories for dependencies
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  include
)

# Add libraries
add_library(vio src/vio.cpp src/frame.cpp src/visual_point.cpp)
add_library(lio src/voxel_map.cpp)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp src/replay_payload_journal.cpp)
target_link_libraries(laser_mapping OpenSSL::Crypto)

# Add the main executable
add_executable(fastlivo_mapping src/main.cpp)

# Deterministic PointCloud2 corruption proxy used for counterfactual replay
# experiments. It intentionally stays separate from the mapper so the clean bag
# remains immutable and baseline/recovery runs receive identical inputs.
add_executable(pointcloud_fault_injector src/pointcloud_fault_injector.cpp)
target_link_libraries(pointcloud_fault_injector ${catkin_LIBRARIES})

# Keeps incumbent outputs live while an isolated recovery generation completes
# a future-stability probation. The canonical topics change generation only
# after timestamp, frame, boundary, divergence-growth, and catch-up checks pass.
add_executable(shadow_output_mux src/shadow_output_mux.cpp)
target_link_libraries(shadow_output_mux ${catkin_LIBRARIES} OpenSSL::Crypto)

# Selects one range-scale hypothesis from isolated candidate health summaries.
# It has no ROS, clean-map, fault-injector, or ground-truth dependency.
add_executable(range_scale_correction_selector
               src/range_scale_correction_selector.cpp)

add_executable(pcd_map_compare src/pcd_map_compare.cpp)
target_link_libraries(pcd_map_compare ${PCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(pcd_map_compare OpenMP::OpenMP_CXX)
endif()

# Offline, read-only forensic export from one sealed recovery handoff. It is
# never launched by the online mapper or public mux.
add_executable(export_recovery_handoff_snapshot
               src/export_recovery_handoff_snapshot.cpp)
target_link_libraries(export_recovery_handoff_snapshot
  laser_mapping vio lio pre imu_proc
  ${catkin_LIBRARIES} ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(export_recovery_handoff_snapshot OpenMP::OpenMP_CXX)
endif()

# Link libraries to the executable
target_link_libraries(fastlivo_mapping
  laser_mapping
  vio
  lio
  pre
  imu_proc
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Link mimalloc if found
if(mimalloc_FOUND)
  target_link_libraries(fastlivo_mapping mimalloc)
endif()


# Curated release tests; original research suite remains in the development workspace.
if(CATKIN_ENABLE_TESTING)
  add_executable(fault_injection_model_test tests/fault_injection_model_test.cpp)
  target_compile_options(fault_injection_model_test PRIVATE -UNDEBUG)
  target_link_libraries(fault_injection_model_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(fault_injection_model_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME fault_injection_model_test COMMAND fault_injection_model_test)
  add_executable(range_scale_correction_estimator_test tests/range_scale_correction_estimator_test.cpp)
  target_compile_options(range_scale_correction_estimator_test PRIVATE -UNDEBUG)
  target_link_libraries(range_scale_correction_estimator_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(range_scale_correction_estimator_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME range_scale_correction_estimator_test COMMAND range_scale_correction_estimator_test)
  add_executable(map_transaction_state_test tests/map_transaction_state_test.cpp)
  target_compile_options(map_transaction_state_test PRIVATE -UNDEBUG)
  target_link_libraries(map_transaction_state_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(map_transaction_state_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME map_transaction_state_test COMMAND map_transaction_state_test)
  add_executable(online_harm_detector_test tests/online_harm_detector_test.cpp)
  target_compile_options(online_harm_detector_test PRIVATE -UNDEBUG)
  target_link_libraries(online_harm_detector_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(online_harm_detector_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME online_harm_detector_test COMMAND online_harm_detector_test)
  add_executable(healthy_tail_validator_test tests/healthy_tail_validator_test.cpp)
  target_compile_options(healthy_tail_validator_test PRIVATE -UNDEBUG)
  target_link_libraries(healthy_tail_validator_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(healthy_tail_validator_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME healthy_tail_validator_test COMMAND healthy_tail_validator_test)
  add_executable(recovery_proposal_arbiter_test tests/recovery_proposal_arbiter_test.cpp)
  target_compile_options(recovery_proposal_arbiter_test PRIVATE -UNDEBUG)
  target_link_libraries(recovery_proposal_arbiter_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(recovery_proposal_arbiter_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME recovery_proposal_arbiter_test COMMAND recovery_proposal_arbiter_test)
  add_executable(recovery_benefit_validator_test tests/recovery_benefit_validator_test.cpp)
  target_compile_options(recovery_benefit_validator_test PRIVATE -UNDEBUG)
  target_link_libraries(recovery_benefit_validator_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(recovery_benefit_validator_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME recovery_benefit_validator_test COMMAND recovery_benefit_validator_test)
  add_executable(selective_residual_gate_test tests/selective_residual_gate_test.cpp)
  target_compile_options(selective_residual_gate_test PRIVATE -UNDEBUG)
  target_link_libraries(selective_residual_gate_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(selective_residual_gate_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME selective_residual_gate_test COMMAND selective_residual_gate_test)
  add_executable(robust_residual_weight_test tests/robust_residual_weight_test.cpp)
  target_compile_options(robust_residual_weight_test PRIVATE -UNDEBUG)
  target_link_libraries(robust_residual_weight_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(robust_residual_weight_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME robust_residual_weight_test COMMAND robust_residual_weight_test)
  add_executable(shadow_generation_selector_test tests/shadow_generation_selector_test.cpp)
  target_compile_options(shadow_generation_selector_test PRIVATE -UNDEBUG)
  target_link_libraries(shadow_generation_selector_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(shadow_generation_selector_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME shadow_generation_selector_test COMMAND shadow_generation_selector_test)
  add_executable(voxel_map_snapshot_test tests/voxel_map_snapshot_test.cpp)
  target_compile_options(voxel_map_snapshot_test PRIVATE -UNDEBUG)
  target_link_libraries(voxel_map_snapshot_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(voxel_map_snapshot_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME voxel_map_snapshot_test COMMAND voxel_map_snapshot_test)
  add_executable(voxel_map_mutation_journal_test tests/voxel_map_mutation_journal_test.cpp)
  target_compile_options(voxel_map_mutation_journal_test PRIVATE -UNDEBUG)
  target_link_libraries(voxel_map_mutation_journal_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(voxel_map_mutation_journal_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME voxel_map_mutation_journal_test COMMAND voxel_map_mutation_journal_test)
  add_executable(raw_lidar_reprocess_test tests/raw_lidar_reprocess_test.cpp)
  target_compile_options(raw_lidar_reprocess_test PRIVATE -UNDEBUG)
  target_link_libraries(raw_lidar_reprocess_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(raw_lidar_reprocess_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME raw_lidar_reprocess_test COMMAND raw_lidar_reprocess_test)
  add_executable(vio_checkpoint_test tests/vio_checkpoint_test.cpp)
  target_compile_options(vio_checkpoint_test PRIVATE -UNDEBUG)
  target_link_libraries(vio_checkpoint_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(vio_checkpoint_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME vio_checkpoint_test COMMAND vio_checkpoint_test)
  add_executable(imu_checkpoint_serialization_test tests/imu_checkpoint_serialization_test.cpp)
  target_compile_options(imu_checkpoint_serialization_test PRIVATE -UNDEBUG)
  target_link_libraries(imu_checkpoint_serialization_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(imu_checkpoint_serialization_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME imu_checkpoint_serialization_test COMMAND imu_checkpoint_serialization_test)
  add_executable(mapper_checkpoint_serialization_test tests/mapper_checkpoint_serialization_test.cpp)
  target_compile_options(mapper_checkpoint_serialization_test PRIVATE -UNDEBUG)
  target_link_libraries(mapper_checkpoint_serialization_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(mapper_checkpoint_serialization_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME mapper_checkpoint_serialization_test COMMAND mapper_checkpoint_serialization_test)
  add_executable(recovery_handoff_bundle_test tests/recovery_handoff_bundle_test.cpp)
  target_compile_options(recovery_handoff_bundle_test PRIVATE -UNDEBUG)
  target_link_libraries(recovery_handoff_bundle_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(recovery_handoff_bundle_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME recovery_handoff_bundle_test COMMAND recovery_handoff_bundle_test)
  add_executable(replay_payload_journal_test tests/replay_payload_journal_test.cpp)
  target_compile_options(replay_payload_journal_test PRIVATE -UNDEBUG)
  target_link_libraries(replay_payload_journal_test laser_mapping vio lio pre imu_proc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Sophus_LIBRARIES} ${Boost_LIBRARIES} OpenSSL::Crypto)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(replay_payload_journal_test OpenMP::OpenMP_CXX)
  endif()
  add_test(NAME replay_payload_journal_test COMMAND replay_payload_journal_test)
endif()
