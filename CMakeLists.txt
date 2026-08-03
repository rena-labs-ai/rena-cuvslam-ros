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

# ---- locate the cuVSLAM SDK (cuvslam2.h + libcuvslam.so) --------------------
# Defaults assume the prebuilt NVIDIA SDK the ansible `cuvslam` role unpacks
# under /opt/cuvslam. Override with -D... if the SDK lives elsewhere.
set(CUVSLAM_SDK_DIR "/opt/cuvslam/17.0.0" CACHE PATH "cuVSLAM SDK root")

# find_* results are cached; when the SDK root changes under an existing build
# tree (e.g. a version bump moves the default), drop the derived entries so
# they re-resolve against the new root instead of silently keeping the old SDK.
if(DEFINED CACHE{_CUVSLAM_SDK_DIR_PREV}
   AND NOT "$CACHE{_CUVSLAM_SDK_DIR_PREV}" STREQUAL "${CUVSLAM_SDK_DIR}")
  unset(CUVSLAM_INCLUDE_DIR CACHE)
  unset(CUVSLAM_LIBRARY CACHE)
endif()
set(_CUVSLAM_SDK_DIR_PREV "${CUVSLAM_SDK_DIR}" CACHE INTERNAL
    "SDK root used at last configure")

set(CUVSLAM_INCLUDE_DIR "${CUVSLAM_SDK_DIR}/include/cuvslam"
    CACHE PATH "Directory containing cuvslam2.h")

# NO_DEFAULT_PATH: the sourced ROS env puts /opt/ros/humble/lib on the search
# path, which carries an incompatible Isaac ROS libcuvslam.so.
find_library(CUVSLAM_LIBRARY
  NAMES cuvslam
  HINTS "${CUVSLAM_SDK_DIR}/bin"
  NO_DEFAULT_PATH)

if(NOT EXISTS "${CUVSLAM_INCLUDE_DIR}/cuvslam2.h")
  message(FATAL_ERROR
    "cuvslam2.h not found in CUVSLAM_INCLUDE_DIR='${CUVSLAM_INCLUDE_DIR}'. "
    "Install the SDK (ansible cuvslam role) or pass -DCUVSLAM_SDK_DIR=<root>.")
endif()
if(NOT CUVSLAM_LIBRARY)
  message(FATAL_ERROR
    "libcuvslam.so not found under '${CUVSLAM_SDK_DIR}/bin'. "
    "Install the SDK (ansible cuvslam role) or pass -DCUVSLAM_SDK_DIR=<root>.")
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
  src/tracker_common.cpp
  src/rgbd_tracker.cpp
  src/stereo_tracker.cpp)

target_include_directories(vslam_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${CUVSLAM_INCLUDE_DIR})

ament_target_dependencies(vslam_node
  rclcpp nav_msgs geometry_msgs sensor_msgs message_filters builtin_interfaces tf2 tf2_ros)

target_link_libraries(vslam_node ${CUVSLAM_LIBRARY} yaml-cpp)

# Ship the SDK's libcuvslam.so *with* the package: installed next to the
# binary so the install tree is self-contained even if /opt/cuvslam moves.
get_filename_component(CUVSLAM_LIB_DIR "${CUVSLAM_LIBRARY}" DIRECTORY)
install(FILES "${CUVSLAM_LIBRARY}" DESTINATION lib/${PROJECT_NAME})

# rpath: at build time load from the SDK dir; once installed, load the
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
