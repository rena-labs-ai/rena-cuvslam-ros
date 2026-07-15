#include "rena_cuvslam_ros/stereo_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <stdexcept>

#include "cuvslam2.h"

using namespace std::chrono_literals;

namespace rena_cuvslam {
namespace {

constexpr int64_t kSlopNs = 5'000'000;  // SLOP_SEC = 0.005 s (same as RGBD)
constexpr int kSyncQueue = 10;          // per-topic buffer for ApproximateTime
constexpr double kCameraInfoTimeoutS = 30.0;

inline int64_t stamp_ns(const sensor_msgs::msg::Image& m) {
  return static_cast<int64_t>(m.header.stamp.sec) * 1'000'000'000 +
         m.header.stamp.nanosec;
}

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Fill a cuVSLAM mono Image (zero-copy) from a ROS MONO8 message.
// Returns false if the encoding is not MONO8 / 8UC1.
bool fill_mono_image(cuvslam::Image& img, const sensor_msgs::msg::Image& msg,
                     int64_t ts, uint32_t cam_index) {
  const std::string enc = to_lower(msg.encoding);
  if (enc != "mono8" && enc != "8uc1") return false;
  img.encoding = cuvslam::ImageData::Encoding::MONO;
  img.data_type = cuvslam::ImageData::DataType::UINT8;
  img.pixels = msg.data.data();
  img.width = static_cast<int32_t>(msg.width);
  img.height = static_cast<int32_t>(msg.height);
  img.pitch = 0;  // ignored for CPU images
  img.is_gpu_mem = false;
  img.timestamp_ns = ts;
  img.camera_index = cam_index;
  return true;
}

}  // namespace

// ------------------------------- StereoTracker ------------------------------

StereoTracker::StereoTracker(rclcpp::Node::SharedPtr node, bool debug)
    : node_(std::move(node)), debug_(debug) {}

StereoTracker::~StereoTracker() { shutdown(); }

void StereoTracker::load_config() {
  for (auto& cam : load_base_oak_cameras()) {
    // cuVSLAM tracks the raw pair, so it needs the true L<->R rigid transform
    // (baseline + small mount rotation), not just a scalar baseline.
    if (!cam.has_stereo_extrinsic) {
      throw std::runtime_error(
          "StereoTracker: OAK " + cam.serial_no + " (base/" + cam.key +
          ") is missing its stereo_extrinsic block (right_from_left) in "
          "/etc/rena/config.yaml, required for the raw stereo baseline");
    }
    const std::string ns = "/base/" + cam.key;
    entries_.push_back(StereoEntry{std::move(cam),
                                   ns + "/left/image_raw",
                                   ns + "/right/image_raw",
                                   ns + "/left/camera_info",
                                   ns + "/right/camera_info"});
  }

  const int n = static_cast<int>(entries_.size());
  RCLCPP_INFO(node_->get_logger(), "[%s] mode=%s  N=%d", tag_.c_str(),
              n > 1 ? "MULTI" : "SINGLE", n);
  for (int i = 0; i < n; ++i) {
    const auto& e = entries_[i];
    RCLCPP_INFO(node_->get_logger(),
                "  cam%d: serial=%s base/%s\n    left:  %s\n    right: %s", i,
                e.serial_no.c_str(), e.key.c_str(),
                e.left_topic.c_str(), e.right_topic.c_str());
  }
}

void StereoTracker::wait_for_camera_info() {
  const int n = static_cast<int>(entries_.size());
  // 2 infos per OAK: [left_0, right_0, left_1, right_1, ...]
  camera_infos_.assign(2 * n, sensor_msgs::msg::CameraInfo());
  std::vector<bool> got(2 * n, false);

  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> info_subs;
  for (int i = 0; i < n; ++i) {
    // Left
    RCLCPP_INFO(node_->get_logger(), "[%s] Waiting for CameraInfo on %s ...",
                tag_.c_str(), entries_[i].left_info_topic.c_str());
    info_subs.push_back(
        node_->create_subscription<sensor_msgs::msg::CameraInfo>(
            entries_[i].left_info_topic, rclcpp::QoS(10),
            [this, i, &got](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
              if (!got[2 * i]) {
                camera_infos_[2 * i] = *msg;
                got[2 * i] = true;
              }
            }));
    // Right
    RCLCPP_INFO(node_->get_logger(), "[%s] Waiting for CameraInfo on %s ...",
                tag_.c_str(), entries_[i].right_info_topic.c_str());
    info_subs.push_back(
        node_->create_subscription<sensor_msgs::msg::CameraInfo>(
            entries_[i].right_info_topic, rclcpp::QoS(10),
            [this, i, &got](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
              if (!got[2 * i + 1]) {
                camera_infos_[2 * i + 1] = *msg;
                got[2 * i + 1] = true;
              }
            }));
  }

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(kCameraInfoTimeoutS);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
    if (std::all_of(got.begin(), got.end(), [](bool b) { return b; })) break;
    rclcpp::spin_some(node_);
    std::this_thread::sleep_for(10ms);
  }
  info_subs.clear();  // unsubscribe

  std::string missing;
  for (int i = 0; i < n; ++i) {
    if (!got[2 * i]) missing += " cam" + std::to_string(i) + "_left";
    if (!got[2 * i + 1]) missing += " cam" + std::to_string(i) + "_right";
  }
  if (!missing.empty()) {
    throw std::runtime_error("Did not receive CameraInfo for" + missing +
                             " within 30 s");
  }
  RCLCPP_INFO(node_->get_logger(),
              "[%s] CameraInfo received for all cameras (left+right)",
              tag_.c_str());
}

void StereoTracker::build_rig_and_tracker() {
  cuvslam::Rig rig;
  // Rig camera order: [l0, r0, l1, r1, ...]
  for (size_t i = 0; i < entries_.size(); ++i) {
    const auto& e = entries_[i];
    const auto& left_info = camera_infos_[2 * i];
    const auto& right_info = camera_infos_[2 * i + 1];

    const RigFromCamera rfc_left = rig_from_camera_from_robot_pose(
        e.roll_deg, e.pitch_deg, e.yaw_deg, e.translation);
    const RigFromCamera rfc_right = rig_from_right_given_left(
        rfc_left, e.right_from_left_rot, e.right_from_left_trans);

    // Helper to fill a cuvslam::Camera from CameraInfo + rig extrinsic.
    auto make_cam = [](const sensor_msgs::msg::CameraInfo& info,
                       const RigFromCamera& rfc) {
      cuvslam::Camera cam;
      cam.size = {static_cast<int32_t>(info.width),
                  static_cast<int32_t>(info.height)};
      cam.focal = {static_cast<float>(info.k[0]), static_cast<float>(info.k[4])};
      cam.principal = {static_cast<float>(info.k[2]),
                       static_cast<float>(info.k[5])};
      cam.distortion.model = cuvslam::Distortion::Model::Polynomial;
      cam.distortion.parameters.resize(8);
      for (int j = 0; j < 8; ++j)
        cam.distortion.parameters[j] =
            j < static_cast<int>(info.d.size())
                ? static_cast<float>(info.d[j])
                : 0.0f;
      cam.rig_from_camera.rotation = {
          static_cast<float>(rfc.rotation[0]), static_cast<float>(rfc.rotation[1]),
          static_cast<float>(rfc.rotation[2]), static_cast<float>(rfc.rotation[3])};
      cam.rig_from_camera.translation = {
          static_cast<float>(rfc.translation[0]),
          static_cast<float>(rfc.translation[1]),
          static_cast<float>(rfc.translation[2])};
      return cam;
    };

    cuvslam::Camera cam_left = make_cam(left_info, rfc_left);
    cuvslam::Camera cam_right = make_cam(right_info, rfc_right);
    rig.cameras.push_back(cam_left);
    rig.cameras.push_back(cam_right);

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] cam%zu base/%s left: size=%dx%d focal=(%.2f,%.2f) "
        "rig_t=(%.4f,%.4f,%.4f)",
        tag_.c_str(), i, e.key.c_str(),
        cam_left.size[0], cam_left.size[1], cam_left.focal[0], cam_left.focal[1],
        cam_left.rig_from_camera.translation[0],
        cam_left.rig_from_camera.translation[1],
        cam_left.rig_from_camera.translation[2]);
    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] cam%zu base/%s right: size=%dx%d focal=(%.2f,%.2f) "
        "rig_t=(%.4f,%.4f,%.4f)",
        tag_.c_str(), i, e.key.c_str(),
        cam_right.size[0], cam_right.size[1], cam_right.focal[0],
        cam_right.focal[1], cam_right.rig_from_camera.translation[0],
        cam_right.rig_from_camera.translation[1],
        cam_right.rig_from_camera.translation[2]);
  }

  // ---- Odometry config (mirrors RosOakStereoTracker.create_odometry_config) ----
  cuvslam::Odometry::Config ocfg;
  ocfg.odometry_mode = cuvslam::Odometry::OdometryMode::Multicamera;
  ocfg.async_sba = false;
  ocfg.rectified_stereo_camera = false;
  ocfg.enable_observations_export = true;
  ocfg.enable_landmarks_export = true;
  ocfg.enable_final_landmarks_export = false;
  // No RGBD settings for stereo mode.

  try {
    cuvslam::WarmUpGPU();
  } catch (const std::exception& ex) {
    RCLCPP_WARN(node_->get_logger(), "[%s] WarmUpGPU failed (continuing): %s",
                tag_.c_str(), ex.what());
  }

  odom_ = std::make_unique<cuvslam::Odometry>(rig, ocfg);

  // ---- SLAM config ----
  cuvslam::Slam::Config scfg;
  scfg.sync_mode = false;
  scfg.planar_constraints = true;
  slam_ = std::make_unique<cuvslam::Slam>(rig, odom_->GetPrimaryCameras(), scfg);

  RCLCPP_INFO(node_->get_logger(),
              "[%s] cuVSLAM Odometry+Slam created (%zu OAKs, %zu stereo cameras)",
              tag_.c_str(), entries_.size(), rig.cameras.size());
}

void StereoTracker::start_streaming() {
  const int n = static_cast<int>(entries_.size());
  if (n > 2) {
    throw std::runtime_error(
        "rena_cuvslam_ros StereoTracker supports 1 or 2 OAK cameras "
        "(ApproximateTime arity is compile-time); got " + std::to_string(n));
  }
  stats_ = std::make_unique<CameraStatsLogger>(node_->get_logger(), tag_, n, debug_);

  cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions opts;
  opts.callback_group = cb_group_;
  const auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();  // BEST_EFFORT

  // Subscriber order: [l0, r0, l1, r1, ...] — matches rig and MatchedSet order.
  std::string topics_log;
  for (int i = 0; i < n; ++i) {
    const auto& e = entries_[i];
    for (int side = 0; side < 2; ++side) {  // 0 = left, 1 = right
      const std::string& topic = side == 0 ? e.left_topic : e.right_topic;
      auto sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(
          node_, topic, qos, opts);
      sub->registerCallback(std::function<void(const ImageMsg::ConstSharedPtr&)>(
          [this, i, side](const ImageMsg::ConstSharedPtr&) {
            stats_->record_raw(i, /*is_depth=*/side == 1);
          }));
      mf_subs_.push_back(std::move(sub));
      topics_log += " " + topic;
    }
  }

  const rclcpp::Duration slop(0, static_cast<uint32_t>(kSlopNs));
  if (n == 1) {
    sync2_ = std::make_shared<message_filters::Synchronizer<SyncPolicy2>>(
        SyncPolicy2(kSyncQueue), *mf_subs_[0], *mf_subs_[1]);
    sync2_->setMaxIntervalDuration(slop);
    sync2_->registerCallback(std::bind(&StereoTracker::on_stereo2, this,
                                       std::placeholders::_1,
                                       std::placeholders::_2));
  } else {
    sync4_ = std::make_shared<message_filters::Synchronizer<SyncPolicy4>>(
        SyncPolicy4(kSyncQueue), *mf_subs_[0], *mf_subs_[1], *mf_subs_[2],
        *mf_subs_[3]);
    sync4_->setMaxIntervalDuration(slop);
    sync4_->registerCallback(std::bind(&StereoTracker::on_stereo4, this,
                                       std::placeholders::_1,
                                       std::placeholders::_2,
                                       std::placeholders::_3,
                                       std::placeholders::_4));
  }

  running_ = true;
  track_thread_ = std::thread(&StereoTracker::track_loop, this);

  RCLCPP_INFO(node_->get_logger(),
              "[%s] Subscribed (message_filters ApproximateTime, slop=%ldms, "
              "queue=%d, %d stereo pair(s)):%s",
              tag_.c_str(), kSlopNs / 1'000'000, kSyncQueue, n,
              topics_log.c_str());
}

void StereoTracker::on_stereo2(const ImageMsg::ConstSharedPtr& l0,
                                const ImageMsg::ConstSharedPtr& r0) {
  emit_set({l0, r0});
}

void StereoTracker::on_stereo4(const ImageMsg::ConstSharedPtr& l0,
                                const ImageMsg::ConstSharedPtr& r0,
                                const ImageMsg::ConstSharedPtr& l1,
                                const ImageMsg::ConstSharedPtr& r1) {
  emit_set({l0, r0, l1, r1});
}

// One synchronized stereo set from ApproximateTime. Latest-wins to the track
// slot so a slow Track() never stalls intake.
void StereoTracker::emit_set(std::vector<ImageMsg::ConstSharedPtr> imgs) {
  const int64_t ts = stamp_ns(*imgs[0]);
  if (ts <= last_ts_) return;  // monotonic guard
  last_ts_ = ts;
  stats_->record_sync();
  stats_->record_decode();  // zero-copy: "decode" == matched & enqueued
  track_slot_.put(MatchedSet{ts, std::move(imgs)});
}

void StereoTracker::track_loop() {
  const uint32_t n_imgs = static_cast<uint32_t>(entries_.size()) * 2;  // left+right per OAK
  while (running_) {
    MatchedSet set;
    if (!track_slot_.get(set, 500ms)) continue;
    if (!running_) break;

    cuvslam::Odometry::ImageSet images;
    images.reserve(n_imgs);
    bool ok = true;
    for (uint32_t idx = 0; idx < n_imgs; ++idx) {
      cuvslam::Image img;
      if (!fill_mono_image(img, *set.images[idx], set.ts, idx)) {
        RCLCPP_WARN(node_->get_logger(),
                    "[%s] bad mono encoding '%s' on stream %u (expected mono8/8uc1)",
                    tag_.c_str(), set.images[idx]->encoding.c_str(), idx);
        ok = false;
        break;
      }
      images.push_back(img);
    }
    if (!ok) continue;

    const auto t0 = std::chrono::steady_clock::now();
    cuvslam::PoseEstimate pe;
    cuvslam::Pose slam_pose;
    bool have_slam = false;
    try {
      pe = odom_->Track(images, {}, {});
      if (pe.world_from_rig.has_value()) {
        cuvslam::Odometry::State state;
        odom_->GetState(state);
        slam_pose = slam_->Track(state);
        have_slam = true;
      }
    } catch (const std::exception& ex) {
      RCLCPP_WARN(node_->get_logger(), "[%s] Track() threw: %s", tag_.c_str(),
                  ex.what());
      continue;
    }
    const double track_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0)
            .count();
    stats_->record_track(track_ms);

    if (have_slam && on_result_) {
      const Quat q = {slam_pose.rotation[0], slam_pose.rotation[1],
                      slam_pose.rotation[2], slam_pose.rotation[3]};
      const Vec3 t = {slam_pose.translation[0], slam_pose.translation[1],
                      slam_pose.translation[2]};
      on_result_(set.ts, to_robot_frame(q, t));
    }
  }
}

void StereoTracker::initialize() {
  load_config();
  wait_for_camera_info();
  build_rig_and_tracker();
  start_streaming();
}

void StereoTracker::shutdown() {
  if (!running_.exchange(false)) {
    // still tear down threads created before running_ was set
  }
  track_slot_.close();
  if (track_thread_.joinable()) track_thread_.join();
  sync2_.reset();
  sync4_.reset();
  mf_subs_.clear();
  slam_.reset();
  odom_.reset();
}

}  // namespace rena_cuvslam
