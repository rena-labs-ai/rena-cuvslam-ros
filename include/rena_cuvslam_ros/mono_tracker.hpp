// Single-camera cuVSLAM tracker — a calibration A/B probe, not a nav mode.
//
// Drives cuVSLAM in Mono mode on ONE image stream (the first config OAK's left
// camera), so the pose is accurate only up to scale. Its purpose is to isolate
// the intrinsics: raw feeds the distorted image with the polynomial model, rect
// feeds the device-rectified image with a pinhole model, and nothing else
// differs — no baseline, no stereo extrinsic, no inter-camera rig.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "rena_cuvslam_ros/frame_conversions.hpp"
#include "rena_cuvslam_ros/tracker_common.hpp"

// Forward-declare so the heavy cuVSLAM header only lands in the .cpp.
namespace cuvslam {
class Odometry;
class Slam;
}  // namespace cuvslam

namespace rena_cuvslam {

class MonoTracker {
 public:
  MonoTracker(rclcpp::Node::SharedPtr node, bool rectified, bool debug);
  ~MonoTracker();

  void set_result_callback(ResultCallback cb) { on_result_ = std::move(cb); }

  // Parse config -> wait for CameraInfo -> build rig -> create Odometry+Slam
  // -> subscribe + start the track thread. Throws std::runtime_error.
  void initialize();
  void shutdown();

 private:
  using ImageMsg = sensor_msgs::msg::Image;

  struct Frame {
    int64_t ts = 0;
    ImageMsg::ConstSharedPtr image;
  };

  void load_config();
  void wait_for_camera_info();
  void build_rig_and_tracker();
  void start_streaming();
  void track_loop();

  rclcpp::Node::SharedPtr node_;
  bool rectified_;
  bool debug_;
  std::string tag_ = "mono";

  OakCameraConfig camera_;
  std::string image_topic_;
  std::string info_topic_;
  sensor_msgs::msg::CameraInfo camera_info_;

  std::unique_ptr<cuvslam::Odometry> odom_;
  std::unique_ptr<cuvslam::Slam> slam_;

  rclcpp::Subscription<ImageMsg>::SharedPtr image_sub_;
  int64_t last_ts_ = 0;

  LatestSlot<Frame> track_slot_;
  std::thread track_thread_;
  std::atomic<bool> running_{false};

  std::unique_ptr<CameraStatsLogger> stats_;
  ResultCallback on_result_;
};

}  // namespace rena_cuvslam
