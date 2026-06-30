// OAK RGBD cuVSLAM tracker (C++ port of rena_cuvslam_ros.RosOakRGBDTracker).
//
// Drives the rena cuVSLAM C++ API (cuvslam2.h: Odometry + Slam, multi-depth
// ICP) directly — no Python, no pybind, no GIL. Color and depth are fed to
// Track() ZERO-COPY straight from the ROS message buffers (RGB as
// ImageData::Encoding::RGB, depth as MONO UINT16), so there is no per-frame
// decode/convert stage at all.
//
// Ingestion is split so DDS is drained at full camera rate regardless of how
// long Track() takes:
//   - subscription callbacks (executor threads, reentrant group): only stash
//     the latest message per stream + count, then a lightweight approximate
//     matcher emits the freshest synchronized RGB-D set;
//   - a single dedicated track thread pulls the latest matched set (stale sets
//     are dropped, freshest wins) and calls Track().
// Net effect: constant 30 fps of frames reach Track(); the only remaining
// limit is Track()'s own GPU cost.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
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

// One base OAK RGB-D camera from /etc/rena/config.yaml.
struct CameraEntry {
  std::string key;
  std::string serial_no;
  std::string color_topic;
  std::string depth_topic;
  std::string info_topic;
  double roll_deg = 0.0;
  double pitch_deg = 0.0;
  double yaw_deg = 0.0;
  Vec3 translation = {0.0, 0.0, 0.0};
};

class RgbdTracker {
 public:
  RgbdTracker(rclcpp::Node::SharedPtr node, double depth_scale, bool debug);
  ~RgbdTracker();

  void set_result_callback(ResultCallback cb) { on_result_ = std::move(cb); }

  // Parse config -> wait for CameraInfo -> build rig -> create Odometry+Slam ->
  // subscribe + start the track thread. Throws std::runtime_error on failure.
  void initialize();
  void shutdown();

 private:
  using ImageMsg = sensor_msgs::msg::Image;
  // Cross-camera time alignment via message_filters ApproximateTime (Konolige
  // algorithm) — the same matcher the Python node used, replacing the naive
  // latest-only matcher that only matched ~2/s for two free-running OAKs.
  // Compile-time arity: N=1 -> 2 streams, N=2 -> 4 streams.
  using SyncPolicy2 = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;
  using SyncPolicy4 =
      message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg, ImageMsg>;

  struct MatchedSet {
    int64_t ts = 0;
    std::vector<ImageMsg::ConstSharedPtr> colors;
    std::vector<ImageMsg::ConstSharedPtr> depths;
  };

  void load_config();
  void wait_for_camera_info();
  void build_rig_and_tracker();  // needs camera_infos_
  void start_streaming();
  void on_rgbd2(const ImageMsg::ConstSharedPtr& c0, const ImageMsg::ConstSharedPtr& d0);
  void on_rgbd4(const ImageMsg::ConstSharedPtr& c0, const ImageMsg::ConstSharedPtr& d0,
                const ImageMsg::ConstSharedPtr& c1, const ImageMsg::ConstSharedPtr& d1);
  void emit_set(std::vector<ImageMsg::ConstSharedPtr> colors,
                std::vector<ImageMsg::ConstSharedPtr> depths);
  void track_loop();

  rclcpp::Node::SharedPtr node_;
  double depth_scale_;
  bool debug_;
  std::string tag_ = "rgbd";

  std::vector<CameraEntry> entries_;
  std::vector<sensor_msgs::msg::CameraInfo> camera_infos_;

  std::unique_ptr<cuvslam::Odometry> odom_;
  std::unique_ptr<cuvslam::Slam> slam_;

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  std::vector<std::shared_ptr<message_filters::Subscriber<ImageMsg>>> mf_subs_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> sync2_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy4>> sync4_;
  int64_t last_ts_ = 0;

  LatestSlot<MatchedSet> track_slot_;
  std::thread track_thread_;
  std::atomic<bool> running_{false};

  std::unique_ptr<CameraStatsLogger> stats_;
  ResultCallback on_result_;
  bool logged_bgr_warning_ = false;
};

}  // namespace rena_cuvslam
