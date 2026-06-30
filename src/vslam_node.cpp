// ROS2 node: runs cuVSLAM (C++ API) on the base OAK RGB-D camera(s) and
// publishes odometry to /cuvslam/odometry + the odom->base_nav_link TF.
//
// C++ port of rena_cuvslam_ros.node.VslamNode (RGBD path only). Behaviour is a
// drop-in match of the Python node:
//   - raw 6-DOF Odometry on /cuvslam/odometry
//   - odom -> child TF (planarized: yaw only, unless planarize:=false)
//   - static map -> odom
// Stamps are the cuVSLAM pipeline timestamp (synced color stamp), the same
// stamp nvblox looks up at depth time.
#include <cmath>
#include <memory>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include "rena_cuvslam_ros/frame_conversions.hpp"
#include "rena_cuvslam_ros/rgbd_tracker.hpp"
#include "rena_cuvslam_ros/stereo_tracker.hpp"

namespace {
constexpr char kOdomTopic[] = "/cuvslam/odometry";
constexpr char kOdomFrame[] = "odom";
}  // namespace

namespace rena_cuvslam {

class VslamNode : public rclcpp::Node {
 public:
  VslamNode() : rclcpp::Node("vslam") {
    child_frame_ = declare_parameter<std::string>("odom_child_frame", "base_nav_link");
    planarize_ = declare_parameter<bool>("planarize", true);
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    debug_ = declare_parameter<bool>("debug", false);
    depth_scale_ = declare_parameter<double>("depth_scale", 0.001);
    declare_parameter<std::string>("tracker", "rgbd");

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(kOdomTopic, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
  }

  // Two-phase init: the constructor can't use shared_from_this(), and the
  // tracker needs the node as a shared_ptr (subscriptions + camera-info wait).
  void start() {
    geometry_msgs::msg::TransformStamped st;
    st.header.stamp = now();
    st.header.frame_id = map_frame_;
    st.child_frame_id = kOdomFrame;
    st.transform.rotation.w = 1.0;
    static_broadcaster_->sendTransform(st);

    const std::string tracker = get_parameter("tracker").as_string();
    if (tracker == "stereo") {
      stereo_tracker_ = std::make_unique<StereoTracker>(shared_from_this(), debug_);
      stereo_tracker_->set_result_callback(
          [this](int64_t ts, const RosPose& pose) { publish(ts, pose); });
      stereo_tracker_->initialize();
    } else {
      if (tracker != "rgbd") {
        RCLCPP_WARN(get_logger(),
                    "Unknown tracker='%s'; falling back to rgbd",
                    tracker.c_str());
      }
      tracker_ = std::make_unique<RgbdTracker>(shared_from_this(), depth_scale_, debug_);
      tracker_->set_result_callback(
          [this](int64_t ts, const RosPose& pose) { publish(ts, pose); });
      tracker_->initialize();
    }

    const std::string mode = planarize_ ? "PLANAR (yaw only)" : "full 6-DOF";
    RCLCPP_INFO(get_logger(),
                "Publishing odometry on %s (raw 6-DOF) + TF %s->%s [%s]; static %s->%s "
                "(tracker=%s)",
                kOdomTopic, kOdomFrame, child_frame_.c_str(), mode.c_str(),
                map_frame_.c_str(), kOdomFrame, tracker.c_str());
  }

  void stop() {
    if (stereo_tracker_) stereo_tracker_->shutdown();
    if (tracker_) tracker_->shutdown();
  }

 private:
  void publish(int64_t ts_ns, const RosPose& pose) {
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(ts_ns / 1'000'000'000);
    stamp.nanosec = static_cast<uint32_t>(ts_ns % 1'000'000'000);

    const auto& t = pose.translation;
    const auto& r = pose.rotation;  // (x, y, z, w)

    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = kOdomFrame;
    msg.child_frame_id = child_frame_;
    msg.pose.pose.position.x = t[0];
    msg.pose.pose.position.y = t[1];
    msg.pose.pose.position.z = t[2];
    msg.pose.pose.orientation.x = r[0];
    msg.pose.pose.orientation.y = r[1];
    msg.pose.pose.orientation.z = r[2];
    msg.pose.pose.orientation.w = r[3];
    odom_pub_->publish(msg);

    // Same pose as TF at the same stamp; topic stays raw 6-DOF, only the TF is
    // planarized (zero VSLAM roll/pitch -- mount tilt lives in the rig TF).
    geometry_msgs::msg::TransformStamped ts;
    ts.header.stamp = stamp;
    ts.header.frame_id = kOdomFrame;
    ts.child_frame_id = child_frame_;
    ts.transform.translation.x = t[0];
    ts.transform.translation.y = t[1];
    ts.transform.translation.z = t[2];
    if (planarize_) {
      const double half = 0.5 * quat_to_yaw(r);
      ts.transform.rotation.z = std::sin(half);
      ts.transform.rotation.w = std::cos(half);
    } else {
      ts.transform.rotation.x = r[0];
      ts.transform.rotation.y = r[1];
      ts.transform.rotation.z = r[2];
      ts.transform.rotation.w = r[3];
    }
    tf_broadcaster_->sendTransform(ts);
  }

  std::string child_frame_;
  std::string map_frame_;
  bool planarize_ = true;
  bool debug_ = false;
  double depth_scale_ = 0.001;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  std::unique_ptr<RgbdTracker> tracker_;
  std::unique_ptr<StereoTracker> stereo_tracker_;
};

}  // namespace rena_cuvslam

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rena_cuvslam::VslamNode>();
  try {
    node->start();
  } catch (const std::exception& ex) {
    RCLCPP_FATAL(node->get_logger(), "tracker init failed: %s", ex.what());
    rclcpp::shutdown();
    return 1;
  }

  // MultiThreadedExecutor + the tracker's Reentrant callback group drains every
  // camera stream concurrently (no GIL, no single-thread head-of-line stall).
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  node->stop();
  rclcpp::shutdown();
  return 0;
}
