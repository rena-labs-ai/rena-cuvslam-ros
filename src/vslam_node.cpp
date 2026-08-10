// ROS2 node: runs cuVSLAM (C++ API) on the base OAK camera(s) and publishes
// the REP-105 split:
//   - raw 6-DOF VO Odometry on /cuvslam/odometry
//   - odom -> child TF from the frontend (VO): continuous, drifts
//   - map -> odom TF from the backend correction (slam ∘ vo⁻¹): jumps on
//     loop closure, so map -> child composes to the backend pose
// TFs are true planar (x, y, yaw only) unless planarize:=false.
// Stamps are the cuVSLAM pipeline timestamp (synced color stamp), the same
// stamp nvblox looks up at depth time.
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
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
    declare_parameter<std::string>("stereo_input", "raw");

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(kOdomTopic, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

  // Two-phase init: the constructor can't use shared_from_this(), and the
  // tracker needs the node as a shared_ptr (subscriptions + camera-info wait).
  void start() {
    const std::string tracker = get_parameter("tracker").as_string();
    if (tracker == "stereo") {
      const std::string input = get_parameter("stereo_input").as_string();
      if (input != "raw" && input != "rect") {
        throw std::runtime_error("stereo_input must be 'raw' or 'rect', got '" +
                                 input + "'");
      }
    }
    if (tracker == "stereo") {
      const std::string input = get_parameter("stereo_input").as_string();
      stereo_tracker_ = std::make_unique<StereoTracker>(shared_from_this(),
                                                        input == "rect", debug_);
      stereo_tracker_->set_result_callback(
          [this](int64_t ts, const RosPose& vo, const RosPose& slam) {
            publish(ts, vo, slam);
          });
      stereo_tracker_->initialize();
    } else {
      if (tracker != "rgbd") {
        RCLCPP_WARN(get_logger(),
                    "Unknown tracker='%s'; falling back to rgbd",
                    tracker.c_str());
      }
      tracker_ = std::make_unique<RgbdTracker>(shared_from_this(), depth_scale_, debug_);
      tracker_->set_result_callback(
          [this](int64_t ts, const RosPose& vo, const RosPose& slam) {
            publish(ts, vo, slam);
          });
      tracker_->initialize();
    }

    const std::string mode = planarize_ ? "PLANAR (yaw only)" : "full 6-DOF";
    RCLCPP_INFO(get_logger(),
                "Publishing VO on %s (raw 6-DOF) + TF %s->%s (frontend) and "
                "%s->%s (backend correction) [%s] (tracker=%s)",
                kOdomTopic, kOdomFrame, child_frame_.c_str(),
                map_frame_.c_str(), kOdomFrame, mode.c_str(), tracker.c_str());
  }

  void stop() {
    if (stereo_tracker_) stereo_tracker_->shutdown();
    if (tracker_) tracker_->shutdown();
  }

 private:
  // True planar pose (x, y, yaw only): roll/pitch/z zeroed. The robot runs on
  // the floor and camera height lives in the static rig TF, so passing VSLAM z
  // through would only inject drift (e.g. floor rising into the nvblox slice).
  static RosPose planar(const RosPose& p) {
    const double half = 0.5 * quat_to_yaw(p.rotation);
    return {{p.translation[0], p.translation[1], 0.0},
            {0.0, 0.0, std::sin(half), std::cos(half)}};
  }

  static geometry_msgs::msg::TransformStamped to_tf(
      const builtin_interfaces::msg::Time& stamp, const std::string& parent,
      const std::string& child, const Vec3& t, const Quat& q) {
    geometry_msgs::msg::TransformStamped ts;
    ts.header.stamp = stamp;
    ts.header.frame_id = parent;
    ts.child_frame_id = child;
    ts.transform.translation.x = t[0];
    ts.transform.translation.y = t[1];
    ts.transform.translation.z = t[2];
    ts.transform.rotation.x = q[0];
    ts.transform.rotation.y = q[1];
    ts.transform.rotation.z = q[2];
    ts.transform.rotation.w = q[3];
    return ts;
  }

  void publish(int64_t ts_ns, const RosPose& vo, const RosPose& slam) {
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(ts_ns / 1'000'000'000);
    stamp.nanosec = static_cast<uint32_t>(ts_ns % 1'000'000'000);

    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = kOdomFrame;
    msg.child_frame_id = child_frame_;
    msg.pose.pose.position.x = vo.translation[0];
    msg.pose.pose.position.y = vo.translation[1];
    msg.pose.pose.position.z = vo.translation[2];
    msg.pose.pose.orientation.x = vo.rotation[0];
    msg.pose.pose.orientation.y = vo.rotation[1];
    msg.pose.pose.orientation.z = vo.rotation[2];
    msg.pose.pose.orientation.w = vo.rotation[3];
    odom_pub_->publish(msg);

    // Topic stays raw 6-DOF VO; only the TFs are planarized.
    const RosPose odom_from_base = planarize_ ? planar(vo) : vo;
    const RosPose map_from_base = planarize_ ? planar(slam) : slam;

    // map->odom = map_from_base ∘ base_from_odom, so the chain composes back
    // to the backend pose: R = R_s R_vᵀ, t = t_s − R t_v.
    const Mat3 r_corr = mat3_mul(quat_to_rotmat(map_from_base.rotation),
                                 mat3_transpose(quat_to_rotmat(odom_from_base.rotation)));
    const Vec3 rotated = mat3_apply(r_corr, odom_from_base.translation);
    const Vec3 t_corr = {map_from_base.translation[0] - rotated[0],
                         map_from_base.translation[1] - rotated[1],
                         map_from_base.translation[2] - rotated[2]};

    tf_broadcaster_->sendTransform(
        {to_tf(stamp, map_frame_, kOdomFrame, t_corr, rotmat_to_quat(r_corr)),
         to_tf(stamp, kOdomFrame, child_frame_, odom_from_base.translation,
               odom_from_base.rotation)});
  }

  std::string child_frame_;
  std::string map_frame_;
  bool planarize_ = true;
  bool debug_ = false;
  double depth_scale_ = 0.001;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
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
