// OAK stereo cuVSLAM tracker (C++ port of rena_cuvslam_ros.RosOakStereoTracker).
//
// Drives the cuVSLAM C++ API in Stereo mode — no depths, left+right MONO8
// images per OAK camera. Reads ALL OAK cameras (any robot_part) from
// /etc/rena/config.yaml, whereas RgbdTracker reads only the base cameras.
//
// Rig layout per OAK i: [left_i, right_i] → cameras order [l0,r0,l1,r1,...].
// right_from_left extrinsic comes from stereo_extrinsic in config.yaml.
//
// Ingestion uses the same LatestSlot + dedicated track thread split as
// RgbdTracker so DDS is drained at full camera rate independently of Track().
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

// One OAK camera (any robot_part) from /etc/rena/config.yaml.
struct StereoEntry {
  std::string key;
  std::string serial_no;
  std::string robot_part;
  std::string left_topic;
  std::string right_topic;
  std::string left_info_topic;
  std::string right_info_topic;
  double roll_deg = 0.0;
  double pitch_deg = 0.0;
  double yaw_deg = 0.0;
  Vec3 translation = {0.0, 0.0, 0.0};
  Quat right_from_left_rot = {0.0, 0.0, 0.0, 1.0};  // (x, y, z, w)
  Vec3 right_from_left_trans = {0.0, 0.0, 0.0};
};

class StereoTracker {
 public:
  StereoTracker(rclcpp::Node::SharedPtr node, bool debug);
  ~StereoTracker();

  void set_result_callback(ResultCallback cb) { on_result_ = std::move(cb); }

  // Parse config -> wait for CameraInfo (left + right per OAK) -> build rig
  // -> create Odometry+Slam -> subscribe + start the track thread.
  // Throws std::runtime_error on failure.
  void initialize();
  void shutdown();

 private:
  using ImageMsg = sensor_msgs::msg::Image;
  // Same compile-time arity approach as RgbdTracker:
  //   N=1 OAK → SyncPolicy2(l0, r0)
  //   N=2 OAKs → SyncPolicy4(l0, r0, l1, r1)
  using SyncPolicy2 =
      message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;
  using SyncPolicy4 =
      message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg,
                                                       ImageMsg, ImageMsg>;

  struct MatchedSet {
    int64_t ts = 0;
    // Interleaved: [l0, r0, l1, r1, ...] — matches rig camera order.
    std::vector<ImageMsg::ConstSharedPtr> images;
  };

  void load_config();
  void wait_for_camera_info();
  void build_rig_and_tracker();
  void start_streaming();
  void on_stereo2(const ImageMsg::ConstSharedPtr& l0,
                  const ImageMsg::ConstSharedPtr& r0);
  void on_stereo4(const ImageMsg::ConstSharedPtr& l0,
                  const ImageMsg::ConstSharedPtr& r0,
                  const ImageMsg::ConstSharedPtr& l1,
                  const ImageMsg::ConstSharedPtr& r1);
  void emit_set(std::vector<ImageMsg::ConstSharedPtr> images);
  void track_loop();

  rclcpp::Node::SharedPtr node_;
  bool debug_;
  std::string tag_ = "stereo";

  std::vector<StereoEntry> entries_;
  // 2 per OAK: camera_infos_[2*i] = left, camera_infos_[2*i+1] = right.
  std::vector<sensor_msgs::msg::CameraInfo> camera_infos_;

  std::unique_ptr<cuvslam::Odometry> odom_;
  std::unique_ptr<cuvslam::Slam> slam_;

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  // Ordered [l0, r0, l1, r1, ...] — matches rig camera order.
  std::vector<std::shared_ptr<message_filters::Subscriber<ImageMsg>>> mf_subs_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> sync2_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy4>> sync4_;
  int64_t last_ts_ = 0;

  LatestSlot<MatchedSet> track_slot_;
  std::thread track_thread_;
  std::atomic<bool> running_{false};

  std::unique_ptr<CameraStatsLogger> stats_;
  ResultCallback on_result_;
};

}  // namespace rena_cuvslam
