#include "rena_cuvslam_ros/rgbd_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "cuvslam2.h"

using namespace std::chrono_literals;

namespace rena_cuvslam {
namespace {

constexpr char kConfigPath[] = "/etc/rena/config.yaml";
constexpr int64_t kSlopNs = 66'000'000;  // SLOP_SEC = 0.066 s
constexpr int kSyncQueue = 10;   // per-topic buffer for ApproximateTime matching
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

// Fill a cuVSLAM color Image (zero-copy) from a ROS color message.
// Returns false if the encoding is not a supported color encoding.
bool fill_color_image(cuvslam::Image& img, const sensor_msgs::msg::Image& msg,
                      int64_t ts, uint32_t cam_index, bool& bgr_seen) {
  const std::string enc = to_lower(msg.encoding);
  if (enc == "rgb8") {
    img.encoding = cuvslam::ImageData::Encoding::RGB;
  } else if (enc == "bgr8") {
    // cuVSLAM has no BGR enum; it converts RGB->gray internally. Feeding BGR
    // bytes only swaps the R/B luma weights, which is harmless for feature
    // tracking (the gray image stays self-consistent frame to frame).
    img.encoding = cuvslam::ImageData::Encoding::RGB;
    bgr_seen = true;
  } else if (enc == "mono8" || enc == "8uc1") {
    img.encoding = cuvslam::ImageData::Encoding::MONO;
  } else {
    return false;
  }
  img.pixels = msg.data.data();
  img.width = static_cast<int32_t>(msg.width);
  img.height = static_cast<int32_t>(msg.height);
  img.pitch = 0;  // ignored for CPU images
  img.data_type = cuvslam::ImageData::DataType::UINT8;
  img.is_gpu_mem = false;
  img.timestamp_ns = ts;
  img.camera_index = cam_index;
  return true;
}

// Fill a cuVSLAM depth Image (zero-copy) from a ROS uint16 depth message.
bool fill_depth_image(cuvslam::Image& img, const sensor_msgs::msg::Image& msg,
                      int64_t ts, uint32_t cam_index) {
  const std::string enc = to_lower(msg.encoding);
  if (enc != "16uc1" && enc != "mono16") return false;
  img.pixels = msg.data.data();
  img.width = static_cast<int32_t>(msg.width);
  img.height = static_cast<int32_t>(msg.height);
  img.pitch = 0;
  img.encoding = cuvslam::ImageData::Encoding::MONO;
  img.data_type = cuvslam::ImageData::DataType::UINT16;
  img.is_gpu_mem = false;
  img.timestamp_ns = ts;
  img.camera_index = cam_index;
  return true;
}

}  // namespace

// ----------------------------- CameraStatsLogger ---------------------------

void CameraStatsLogger::record_raw(int cam, bool is_depth) {
  std::lock_guard<std::mutex> lk(mu_);
  if (is_depth)
    ++raw_depth_[cam];
  else
    ++raw_rgb_[cam];
  maybe_log_locked();
}

void CameraStatsLogger::record_track(double track_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  ++track_n_;
  if (track_ms > kBudgetMs) {
    ++over_n_;
    over_ms_ += track_ms;
  } else {
    ++on_n_;
  }
}

void CameraStatsLogger::maybe_log_locked() {
  // TEMP: always emit the one-line ladder (INFO, 1/s) regardless of `debug_`,
  // so we don't have to set global log_level:=debug (which spams everything).
  (void)debug_;
  const auto now = std::chrono::steady_clock::now();
  if (now - last_log_ < 1s) return;

  std::string cams;
  for (int i = 0; i < n_; ++i) {
    if (i) cams += " | ";
    cams += "cam" + std::to_string(i) + " rgb " + std::to_string(raw_rgb_[i]) +
            " depth " + std::to_string(raw_depth_[i]);
  }
  RCLCPP_INFO(
      logger_, "[%s] %s | sync %d/s | decode %d/s | track %d/s <=33 %d >33 %d (sum %.0fms)",
      tag_.c_str(), cams.c_str(), sync_n_, decode_n_, track_n_, on_n_, over_n_,
      over_ms_);

  std::fill(raw_rgb_.begin(), raw_rgb_.end(), 0);
  std::fill(raw_depth_.begin(), raw_depth_.end(), 0);
  sync_n_ = decode_n_ = track_n_ = on_n_ = over_n_ = 0;
  over_ms_ = 0.0;
  last_log_ = now;
}

// ------------------------------- RgbdTracker --------------------------------

RgbdTracker::RgbdTracker(rclcpp::Node::SharedPtr node, double depth_scale, bool debug)
    : node_(std::move(node)), depth_scale_(depth_scale), debug_(debug) {}

RgbdTracker::~RgbdTracker() { shutdown(); }

void RgbdTracker::load_config() {
  YAML::Node root = YAML::LoadFile(kConfigPath);
  const YAML::Node base = root["base"];
  if (!base || !base["cameras"]) {
    throw std::runtime_error(
        "RgbdTracker: no base.cameras in " + std::string(kConfigPath));
  }
  for (const auto& cam : base["cameras"]) {
    if (!cam["type"] || cam["type"].as<std::string>() != "oak") continue;
    CameraEntry e;
    e.key = cam["key"] ? cam["key"].as<std::string>() : "";
    e.serial_no = cam["serial_no"] ? cam["serial_no"].as<std::string>() : "";
    const std::string ns = "/base/" + e.key;
    e.color_topic = ns + "/rgb/image_raw";
    e.depth_topic = ns + "/stereo/image_raw";
    e.info_topic = ns + "/rgb/camera_info";
    if (const YAML::Node rig = cam["rig"]) {
      if (const YAML::Node t = rig["translation"]) {
        if (t.size() != 3)
          throw std::runtime_error("rig.translation must have 3 elements for " + e.key);
        e.translation = {t[0].as<double>(), t[1].as<double>(), t[2].as<double>()};
      }
      if (const YAML::Node r = rig["rotation"]) {
        e.roll_deg = r["roll"] ? r["roll"].as<double>() : 0.0;
        e.pitch_deg = r["pitch"] ? r["pitch"].as<double>() : 0.0;
        e.yaw_deg = r["yaw"] ? r["yaw"].as<double>() : 0.0;
      }
    }
    entries_.push_back(std::move(e));
  }
  if (entries_.empty()) {
    throw std::runtime_error(
        "RgbdTracker: no base OAK camera in " + std::string(kConfigPath));
  }

  const int n = static_cast<int>(entries_.size());
  RCLCPP_INFO(node_->get_logger(), "[%s] mode=%s  N=%d", tag_.c_str(),
              n > 1 ? "MULTI" : "SINGLE", n);
  for (int i = 0; i < n; ++i) {
    const auto& e = entries_[i];
    RCLCPP_INFO(node_->get_logger(),
                "  cam%d: serial=%s base/%s\n    color: %s\n    depth: %s", i,
                e.serial_no.c_str(), e.key.c_str(), e.color_topic.c_str(),
                e.depth_topic.c_str());
  }
}

void RgbdTracker::wait_for_camera_info() {
  const int n = static_cast<int>(entries_.size());
  camera_infos_.assign(n, sensor_msgs::msg::CameraInfo());
  std::vector<bool> got(n, false);

  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> info_subs;
  for (int i = 0; i < n; ++i) {
    RCLCPP_INFO(node_->get_logger(), "[%s] Waiting for CameraInfo on %s ...",
                tag_.c_str(), entries_[i].info_topic.c_str());
    info_subs.push_back(node_->create_subscription<sensor_msgs::msg::CameraInfo>(
        entries_[i].info_topic, rclcpp::QoS(10),
        [this, i, &got](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
          if (!got[i]) {
            camera_infos_[i] = *msg;
            got[i] = true;
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
  for (int i = 0; i < n; ++i)
    if (!got[i]) missing += " cam" + std::to_string(i);
  if (!missing.empty()) {
    throw std::runtime_error("Did not receive CameraInfo for" + missing +
                             " within 30 s");
  }
  RCLCPP_INFO(node_->get_logger(), "[%s] CameraInfo received for all cameras",
              tag_.c_str());
}

void RgbdTracker::build_rig_and_tracker() {
  cuvslam::Rig rig;
  for (size_t i = 0; i < entries_.size(); ++i) {
    const auto& e = entries_[i];
    const auto& info = camera_infos_[i];

    const RigFromCamera rfc = rig_from_camera_from_robot_pose(
        e.roll_deg, e.pitch_deg, e.yaw_deg, e.translation);

    cuvslam::Camera cam;
    cam.size = {static_cast<int32_t>(info.width),
                static_cast<int32_t>(info.height)};
    cam.focal = {static_cast<float>(info.k[0]), static_cast<float>(info.k[4])};
    cam.principal = {static_cast<float>(info.k[2]), static_cast<float>(info.k[5])};
    // ROS rational_polynomial D -> cuVSLAM Polynomial = first 8 OpenCV coeffs
    // [k1, k2, p1, p2, k3, k4, k5, k6]; indices 8.. are thin-prism terms the
    // Polynomial model doesn't carry.
    cam.distortion.model = cuvslam::Distortion::Model::Polynomial;
    cam.distortion.parameters.resize(8);
    for (int j = 0; j < 8; ++j)
      cam.distortion.parameters[j] =
          j < static_cast<int>(info.d.size()) ? static_cast<float>(info.d[j]) : 0.0f;
    cam.rig_from_camera.rotation = {
        static_cast<float>(rfc.rotation[0]), static_cast<float>(rfc.rotation[1]),
        static_cast<float>(rfc.rotation[2]), static_cast<float>(rfc.rotation[3])};
    cam.rig_from_camera.translation = {static_cast<float>(rfc.translation[0]),
                                       static_cast<float>(rfc.translation[1]),
                                       static_cast<float>(rfc.translation[2])};
    rig.cameras.push_back(cam);

    RCLCPP_INFO(
        node_->get_logger(),
        "[%s] cam%zu base/%s: size=%dx%d focal=(%.2f,%.2f) principal=(%.2f,%.2f) "
        "rig_t=(%.4f,%.4f,%.4f) rig_q=(%.4f,%.4f,%.4f,%.4f)",
        tag_.c_str(), i, e.key.c_str(), cam.size[0], cam.size[1], cam.focal[0],
        cam.focal[1], cam.principal[0], cam.principal[1],
        cam.rig_from_camera.translation[0], cam.rig_from_camera.translation[1],
        cam.rig_from_camera.translation[2], cam.rig_from_camera.rotation[0],
        cam.rig_from_camera.rotation[1], cam.rig_from_camera.rotation[2],
        cam.rig_from_camera.rotation[3]);
  }

  // ---- Odometry config (mirrors RosOakRGBDTracker.create_odometry_config) ----
  cuvslam::Odometry::Config ocfg;
  ocfg.odometry_mode = cuvslam::Odometry::OdometryMode::RGBD;
  ocfg.async_sba = true;
  ocfg.rectified_stereo_camera = false;
  // SLAM requires observations + landmarks export (cf. the pycuvslam Tracker
  // wrapper, which force-enables both when SLAM is on).
  ocfg.enable_observations_export = true;
  ocfg.enable_landmarks_export = true;
  // NOT enable_final_landmarks_export: it only feeds GetFinalLandmarks() (viz),
  // which this node never calls, and the header notes it slows Track(). The
  // Python config left it on by copy-paste; dropping it is a free latency win
  // that trims the >33 ms Track() tail with zero effect on odometry/SLAM output.
  ocfg.enable_final_landmarks_export = false;

  const float scale_factor = static_cast<float>(1.0 / depth_scale_);
  ocfg.rgbd_settings.enable_depth_stereo_tracking = false;
  if (entries_.size() == 1) {
    ocfg.rgbd_settings.depth_camera_id = 0;
    ocfg.rgbd_settings.depth_scale_factor = scale_factor;
  } else {
    // Multi-camera: one depth source per rig camera (cuVSLAM#3 multi-depth ICP).
    for (size_t i = 0; i < entries_.size(); ++i) {
      cuvslam::Odometry::RGBDSettings::DepthCameraSettings d;
      d.camera_id = static_cast<int32_t>(i);
      d.depth_scale_factor = scale_factor;
      ocfg.rgbd_settings.depth_cameras.push_back(d);
    }
  }

  try {
    cuvslam::WarmUpGPU();
  } catch (const std::exception& ex) {
    RCLCPP_WARN(node_->get_logger(), "[%s] WarmUpGPU failed (continuing): %s",
                tag_.c_str(), ex.what());
  }

  odom_ = std::make_unique<cuvslam::Odometry>(rig, ocfg);

  // ---- SLAM config (mirrors create_slam_config) ----
  cuvslam::Slam::Config scfg;
  scfg.sync_mode = false;
  scfg.planar_constraints = true;
  slam_ = std::make_unique<cuvslam::Slam>(rig, odom_->GetPrimaryCameras(), scfg);

  RCLCPP_INFO(node_->get_logger(), "[%s] cuVSLAM Odometry+Slam created (%zu cameras, %s)",
              tag_.c_str(), entries_.size(),
              entries_.size() > 1 ? "multi-depth ICP" : "single RGBD");
}

void RgbdTracker::start_streaming() {
  const int n = static_cast<int>(entries_.size());
  if (n > 2) {
    throw std::runtime_error(
        "rena_cuvslam_ros supports 1 or 2 base OAK cameras "
        "(ApproximateTime arity is compile-time); got " + std::to_string(n));
  }
  stats_ = std::make_unique<CameraStatsLogger>(node_->get_logger(), tag_, n, debug_);

  // Reentrant group so a MultiThreadedExecutor drains every stream concurrently
  // (no head-of-line blocking). Streams are ordered [c0, d0, c1, d1] so the
  // synchronizer callback args map straight onto camera index.
  cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions opts;
  opts.callback_group = cb_group_;
  const auto qos = rclcpp::SensorDataQoS().get_rmw_qos_profile();  // BEST_EFFORT

  std::string topics_log;
  for (int i = 0; i < n; ++i) {
    const auto& e = entries_[i];
    for (int s = 0; s < 2; ++s) {  // 0 = color, 1 = depth
      const std::string& topic = s == 0 ? e.color_topic : e.depth_topic;
      auto sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(
          node_, topic, qos, opts);
      // Tap the raw stream for the per-camera counter (independent of the sync).
      sub->registerCallback(std::function<void(const ImageMsg::ConstSharedPtr&)>(
          [this, i, s](const ImageMsg::ConstSharedPtr&) {
            stats_->record_raw(i, /*is_depth=*/s == 1);
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
    sync2_->registerCallback(std::bind(&RgbdTracker::on_rgbd2, this,
                                       std::placeholders::_1, std::placeholders::_2));
  } else {
    sync4_ = std::make_shared<message_filters::Synchronizer<SyncPolicy4>>(
        SyncPolicy4(kSyncQueue), *mf_subs_[0], *mf_subs_[1], *mf_subs_[2], *mf_subs_[3]);
    sync4_->setMaxIntervalDuration(slop);
    sync4_->registerCallback(std::bind(&RgbdTracker::on_rgbd4, this,
                                       std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3, std::placeholders::_4));
  }

  running_ = true;
  track_thread_ = std::thread(&RgbdTracker::track_loop, this);

  RCLCPP_INFO(node_->get_logger(),
              "[%s] Subscribed (message_filters ApproximateTime, slop=%ldms, "
              "queue=%d, %d RGB-D pair(s)):%s",
              tag_.c_str(), kSlopNs / 1'000'000, kSyncQueue, n, topics_log.c_str());
}

void RgbdTracker::on_rgbd2(const ImageMsg::ConstSharedPtr& c0,
                           const ImageMsg::ConstSharedPtr& d0) {
  emit_set({c0}, {d0});
}

void RgbdTracker::on_rgbd4(const ImageMsg::ConstSharedPtr& c0,
                           const ImageMsg::ConstSharedPtr& d0,
                           const ImageMsg::ConstSharedPtr& c1,
                           const ImageMsg::ConstSharedPtr& d1) {
  emit_set({c0, c1}, {d0, d1});
}

// One synchronized RGB-D set out of ApproximateTime. Push the freshest to the
// track slot (latest-wins, drop stale) so a slow Track() never stalls intake.
void RgbdTracker::emit_set(std::vector<ImageMsg::ConstSharedPtr> colors,
                           std::vector<ImageMsg::ConstSharedPtr> depths) {
  const int64_t ts = stamp_ns(*colors[0]);
  if (ts <= last_ts_) return;  // monotonic guard
  last_ts_ = ts;
  stats_->record_sync();
  stats_->record_decode();  // zero-copy: "decode" == matched & enqueued
  track_slot_.put(MatchedSet{ts, std::move(colors), std::move(depths)});
}

void RgbdTracker::track_loop() {
  const uint32_t n = static_cast<uint32_t>(entries_.size());
  while (running_) {
    MatchedSet set;
    if (!track_slot_.get(set, 500ms)) continue;
    if (!running_) break;

    cuvslam::Odometry::ImageSet images, depths;
    images.reserve(n);
    depths.reserve(n);
    bool ok = true;
    for (uint32_t i = 0; i < n; ++i) {
      cuvslam::Image color, depth;
      if (!fill_color_image(color, *set.colors[i], set.ts, i, logged_bgr_warning_)) {
        RCLCPP_WARN(node_->get_logger(), "[%s] bad color encoding '%s' on cam%u",
                    tag_.c_str(), set.colors[i]->encoding.c_str(), i);
        ok = false;
        break;
      }
      if (!fill_depth_image(depth, *set.depths[i], set.ts, i)) {
        RCLCPP_WARN(node_->get_logger(), "[%s] bad depth encoding '%s' on cam%u",
                    tag_.c_str(), set.depths[i]->encoding.c_str(), i);
        ok = false;
        break;
      }
      images.push_back(color);
      depths.push_back(depth);
    }
    if (!ok) continue;

    if (logged_bgr_warning_) {
      RCLCPP_WARN_ONCE(node_->get_logger(),
                       "[%s] color stream is bgr8; fed to cuVSLAM as RGB "
                       "(luma weights swapped, negligible for tracking)",
                       tag_.c_str());
    }

    const auto t0 = std::chrono::steady_clock::now();
    cuvslam::PoseEstimate pe;
    cuvslam::Pose slam_pose;
    bool have_slam = false;
    try {
      pe = odom_->Track(images, {}, depths);
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

void RgbdTracker::initialize() {
  load_config();
  wait_for_camera_info();
  build_rig_and_tracker();
  start_streaming();
}

void RgbdTracker::shutdown() {
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
