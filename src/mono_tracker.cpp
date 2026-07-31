#include "rena_cuvslam_ros/mono_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "cuvslam2.h"

using namespace std::chrono_literals;

namespace rena_cuvslam {
namespace {

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

// MONO8/8UC1 is zero-copy; BGR8/RGB8 (the depthai driver replicates the IR
// mono channel to 3 channels) extracts channel 0 into `scratch`, which must
// outlive the Track() call.
bool fill_mono_image(cuvslam::Image& img, const sensor_msgs::msg::Image& msg,
                     int64_t ts, std::vector<uint8_t>& scratch) {
  const std::string enc = to_lower(msg.encoding);
  if (enc == "mono8" || enc == "8uc1") {
    img.pixels = msg.data.data();
  } else if (enc == "bgr8" || enc == "rgb8") {
    scratch.resize(static_cast<size_t>(msg.width) * msg.height);
    for (uint32_t row = 0; row < msg.height; ++row) {
      const uint8_t* src = msg.data.data() + static_cast<size_t>(row) * msg.step;
      uint8_t* dst = scratch.data() + static_cast<size_t>(row) * msg.width;
      for (uint32_t col = 0; col < msg.width; ++col) dst[col] = src[col * 3];
    }
    img.pixels = scratch.data();
  } else {
    return false;
  }
  img.encoding = cuvslam::ImageData::Encoding::MONO;
  img.data_type = cuvslam::ImageData::DataType::UINT8;
  img.width = static_cast<int32_t>(msg.width);
  img.height = static_cast<int32_t>(msg.height);
  img.pitch = 0;  // ignored for CPU images
  img.is_gpu_mem = false;
  img.timestamp_ns = ts;
  img.camera_index = 0;
  return true;
}

}  // namespace

MonoTracker::MonoTracker(rclcpp::Node::SharedPtr node, bool rectified,
                         bool debug)
    : node_(std::move(node)), rectified_(rectified), debug_(debug) {}

MonoTracker::~MonoTracker() { shutdown(); }

void MonoTracker::load_config() {
  auto cams = load_base_oak_cameras();
  if (cams.empty()) {
    throw std::runtime_error("MonoTracker: no base OAK cameras in config");
  }
  camera_ = cams.front();  // probe mode: the first configured OAK only
  const std::string ns = "/base/" + camera_.key;
  if (rectified_) {
    image_topic_ = ns + "/left_rect/image_rect";
    info_topic_ = ns + "/left_rect/camera_info";
  } else {
    image_topic_ = ns + "/left/image_raw";
    info_topic_ = ns + "/left/camera_info";
  }
  RCLCPP_INFO(node_->get_logger(),
              "[%s] input=%s  base/%s (serial=%s)\n    image: %s", tag_.c_str(),
              rectified_ ? "rect" : "raw", camera_.key.c_str(),
              camera_.serial_no.c_str(), image_topic_.c_str());
  if (cams.size() > 1) {
    RCLCPP_INFO(node_->get_logger(),
                "[%s] ignoring %zu other configured OAK(s) — single camera by "
                "design", tag_.c_str(), cams.size() - 1);
  }
}

void MonoTracker::wait_for_camera_info() {
  bool got = false;
  auto sub = node_->create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic_, rclcpp::QoS(10),
      [this, &got](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
        if (!got) {
          camera_info_ = *msg;
          got = true;
        }
      });
  RCLCPP_INFO(node_->get_logger(), "[%s] Waiting for CameraInfo on %s ...",
              tag_.c_str(), info_topic_.c_str());
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(kCameraInfoTimeoutS);
  while (rclcpp::ok() && !got && std::chrono::steady_clock::now() < deadline) {
    rclcpp::spin_some(node_);
    std::this_thread::sleep_for(10ms);
  }
  sub.reset();
  if (!got) {
    throw std::runtime_error("Did not receive CameraInfo on " + info_topic_ +
                             " within 30 s");
  }
}

void MonoTracker::build_rig_and_tracker() {
  cuvslam::Rig rig;

  const RigFromCamera rfc_phys = rig_from_camera_from_robot_pose(
      camera_.roll_deg, camera_.pitch_deg, camera_.yaw_deg, camera_.translation);
  RigFromCamera rfc = rfc_phys;
  if (rectified_) {
    // The rect stream lives in the mesh's virtual camera: the physical camera
    // rotated by the rectification rotation (camera_info.r).
    Mat3 rect_r;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) rect_r[r][c] = camera_info_.r[3 * r + c];
    const Mat3 r_rig_rect = mat3_mul(quat_to_rotmat(rfc_phys.rotation),
                                     mat3_transpose(rect_r));
    rfc = RigFromCamera{rotmat_to_quat(r_rig_rect), rfc_phys.translation};
  }

  cuvslam::Camera cam;
  cam.size = {static_cast<int32_t>(camera_info_.width),
              static_cast<int32_t>(camera_info_.height)};
  cam.focal = {static_cast<float>(camera_info_.k[0]),
               static_cast<float>(camera_info_.k[4])};
  cam.principal = {static_cast<float>(camera_info_.k[2]),
                   static_cast<float>(camera_info_.k[5])};
  if (rectified_) {
    cam.distortion.model = cuvslam::Distortion::Model::Pinhole;
    cam.distortion.parameters.clear();
  } else {
    cam.distortion.model = cuvslam::Distortion::Model::Polynomial;
    cam.distortion.parameters.resize(8);
    for (int j = 0; j < 8; ++j)
      cam.distortion.parameters[j] =
          j < static_cast<int>(camera_info_.d.size())
              ? static_cast<float>(camera_info_.d[j])
              : 0.0f;
  }
  cam.rig_from_camera.rotation = {
      static_cast<float>(rfc.rotation[0]), static_cast<float>(rfc.rotation[1]),
      static_cast<float>(rfc.rotation[2]), static_cast<float>(rfc.rotation[3])};
  cam.rig_from_camera.translation = {static_cast<float>(rfc.translation[0]),
                                     static_cast<float>(rfc.translation[1]),
                                     static_cast<float>(rfc.translation[2])};
  rig.cameras.push_back(cam);

  RCLCPP_INFO(node_->get_logger(),
              "[%s] base/%s: size=%dx%d focal=(%.2f,%.2f) principal=(%.2f,%.2f) "
              "distortion=%s", tag_.c_str(), camera_.key.c_str(), cam.size[0],
              cam.size[1], cam.focal[0], cam.focal[1], cam.principal[0],
              cam.principal[1], rectified_ ? "pinhole" : "polynomial(8)");

  cuvslam::Odometry::Config ocfg;
  ocfg.odometry_mode = cuvslam::Odometry::OdometryMode::Mono;
  ocfg.async_sba = false;
  ocfg.enable_observations_export = true;
  ocfg.enable_landmarks_export = true;
  ocfg.enable_final_landmarks_export = false;

  try {
    cuvslam::WarmUpGPU();
  } catch (const std::exception& ex) {
    RCLCPP_WARN(node_->get_logger(), "[%s] WarmUpGPU failed (continuing): %s",
                tag_.c_str(), ex.what());
  }

  odom_ = std::make_unique<cuvslam::Odometry>(rig, ocfg);

  cuvslam::Slam::Config scfg;
  scfg.sync_mode = false;
  scfg.planar_constraints = true;
  slam_ = std::make_unique<cuvslam::Slam>(rig, odom_->GetPrimaryCameras(), scfg);

  RCLCPP_INFO(node_->get_logger(),
              "[%s] cuVSLAM Odometry+Slam created (Mono — pose is up to scale)",
              tag_.c_str());
}

void MonoTracker::start_streaming() {
  stats_ = std::make_unique<CameraStatsLogger>(
      node_->get_logger(), tag_, 1, debug_,
      std::array<std::string, 2>{"image", "unused"}, /*show_decode=*/false);

  image_sub_ = node_->create_subscription<ImageMsg>(
      image_topic_, rclcpp::SensorDataQoS(),
      [this](ImageMsg::ConstSharedPtr msg) {
        stats_->record_raw(0, 0);
        const int64_t ts = stamp_ns(*msg);
        if (ts <= last_ts_) return;  // monotonic guard
        last_ts_ = ts;
        stats_->record_sync();
        track_slot_.put(Frame{ts, std::move(msg)});
      });

  running_ = true;
  track_thread_ = std::thread(&MonoTracker::track_loop, this);

  RCLCPP_INFO(node_->get_logger(), "[%s] Subscribed: %s", tag_.c_str(),
              image_topic_.c_str());
}

void MonoTracker::track_loop() {
  std::vector<uint8_t> scratch;
  while (running_) {
    Frame frame;
    if (!track_slot_.get(frame, 500ms)) continue;
    if (!running_) break;

    cuvslam::Image img;
    if (!fill_mono_image(img, *frame.image, frame.ts, scratch)) {
      RCLCPP_WARN(node_->get_logger(),
                  "[%s] bad mono encoding '%s' (expected mono8/8uc1/bgr8/rgb8)",
                  tag_.c_str(), frame.image->encoding.c_str());
      continue;
    }

    const auto t0 = std::chrono::steady_clock::now();
    cuvslam::PoseEstimate pe;
    cuvslam::Pose slam_pose;
    bool have_slam = false;
    try {
      pe = odom_->Track({img}, {}, {});
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
    stats_->record_track(std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count());

    if (have_slam && on_result_) {
      const cuvslam::Pose& vo = pe.world_from_rig->pose;
      on_result_(
          frame.ts,
          to_robot_frame({vo.rotation[0], vo.rotation[1], vo.rotation[2],
                          vo.rotation[3]},
                         {vo.translation[0], vo.translation[1],
                          vo.translation[2]}),
          to_robot_frame({slam_pose.rotation[0], slam_pose.rotation[1],
                          slam_pose.rotation[2], slam_pose.rotation[3]},
                         {slam_pose.translation[0], slam_pose.translation[1],
                          slam_pose.translation[2]}));
    }
  }
}

void MonoTracker::initialize() {
  load_config();
  wait_for_camera_info();
  build_rig_and_tracker();
  start_streaming();
}

void MonoTracker::shutdown() {
  running_ = false;
  track_slot_.close();
  if (track_thread_.joinable()) track_thread_.join();
  image_sub_.reset();
  slam_.reset();
  odom_.reset();
}

}  // namespace rena_cuvslam
