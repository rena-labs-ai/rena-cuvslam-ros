cmake_minimum_required(VERSION 3.16)
project(rena_cuvslam_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)  # cuvslam2.h needs std::optional / string_view
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# ---- locate the rena cuVSLAM C++ API (cuvslam2.h + libcuvslam.so) ----------
# Defaults assume the sibling source checkout at ../cuVSLAM with build/bin
# populated by its build_and_install.sh (Stage 1). Override with -D... if the
# library lives elsewhere (e.g. a packaged/installed location).
set(CUVSLAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cuVSLAM"
    CACHE PATH "cuVSLAM source checkout root")
set(CUVSLAM_INCLUDE_DIR "${CUVSLAM_DIR}/libs/cuvslam"
    CACHE PATH "Directory containing cuvslam2.h")

find_library(CUVSLAM_LIBRARY
  NAMES cuvslam
  HINTS "${CUVSLAM_DIR}/build/bin"
  NO_DEFAULT_PATH)
if(NOT CUVSLAM_LIBRARY)
  find_library(CUVSLAM_LIBRARY NAMES cuvslam)  # fall back to system search path
endif()

if(NOT EXISTS "${CUVSLAM_INCLUDE_DIR}/cuvslam2.h")
  message(FATAL_ERROR
    "cuvslam2.h not found in CUVSLAM_INCLUDE_DIR='${CUVSLAM_INCLUDE_DIR}'. "
    "Point -DCUVSLAM_DIR=<cuVSLAM repo root> at your checkout.")
endif()
if(NOT CUVSLAM_LIBRARY)
  message(FATAL_ERROR
    "libcuvslam.so not found. Build it (cuVSLAM/build_and_install.sh Stage 1) "
    "or pass -DCUVSLAM_LIBRARY=/path/to/libcuvslam.so.")
endif()
message(STATUS "cuVSLAM header : ${CUVSLAM_INCLUDE_DIR}/cuvslam2.h")
message(STATUS "cuVSLAM library: ${CUVSLAM_LIBRARY}")

# ---- ROS / system dependencies ---------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(vslam_node
  src/vslam_node.cpp
  src/rgbd_tracker.cpp
  src/stereo_tracker.cpp)

target_include_directories(vslam_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${CUVSLAM_INCLUDE_DIR})

ament_target_dependencies(vslam_node
  rclcpp nav_msgs geometry_msgs sensor_msgs message_filters builtin_interfaces tf2 tf2_ros)

target_link_libraries(vslam_node ${CUVSLAM_LIBRARY} yaml-cpp)

# Ship our cuVSLAM build *with* the package: install libcuvslam.so next to the
# binary so the install tree is self-contained and does not depend on the
# transient ../cuVSLAM/build/bin directory surviving.
get_filename_component(CUVSLAM_LIB_DIR "${CUVSLAM_LIBRARY}" DIRECTORY)
install(FILES "${CUVSLAM_LIBRARY}" DESTINATION lib/${PROJECT_NAME})

# rpath: at build time load from the cuVSLAM build dir; once installed, load the
# co-located copy via $ORIGIN. Crucially, force the legacy DT_RPATH tag
# (--disable-new-dtags) instead of DT_RUNPATH: DT_RPATH is searched *before*
# LD_LIBRARY_PATH, so our libcuvslam.so wins over the incompatible Isaac ROS
# copy in /opt/ros/humble/lib (which the ROS setup puts on LD_LIBRARY_PATH).
set_target_properties(vslam_node PROPERTIES
  BUILD_RPATH "${CUVSLAM_LIB_DIR}"
  INSTALL_RPATH "$ORIGIN;/usr/local/cuda/lib64")
target_link_options(vslam_node PRIVATE "-Wl,--disable-new-dtags")

install(TARGETS vslam_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
