// Shared /etc/rena/config.yaml parsing for the RGBD and Stereo trackers.
#include "rena_cuvslam_ros/tracker_common.hpp"

#include <cstdlib>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace rena_cuvslam {
namespace {
constexpr char kConfigPath[] = "/etc/rena/config.yaml";

// RENA_CONFIG_PATH overrides the robot config, e.g. to replay a bag against a
// reduced camera set without touching /etc/rena/config.yaml.
std::string config_path() {
  const char* env = std::getenv("RENA_CONFIG_PATH");
  return (env && *env) ? env : kConfigPath;
}
}  // namespace

std::vector<OakCameraConfig> load_base_oak_cameras() {
  const std::string path = config_path();
  YAML::Node root = YAML::LoadFile(path);
  const YAML::Node base = root["base"];
  if (!base || !base["cameras"]) {
    throw std::runtime_error("no base.cameras in " + path);
  }

  std::vector<OakCameraConfig> out;
  for (const auto& cam : base["cameras"]) {
    if (!cam["type"] || cam["type"].as<std::string>() != "oak") continue;

    OakCameraConfig c;
    c.key = cam["key"] ? cam["key"].as<std::string>() : "";
    c.serial_no = cam["serial_no"] ? cam["serial_no"].as<std::string>() : "";
    // Topics are derived as /base/<key>/..., so an empty or duplicate key
    // yields malformed topics or two rig entries eating the same streams.
    if (c.key.empty()) {
      throw std::runtime_error("base OAK camera (serial '" + c.serial_no +
                               "') has no key in " + path);
    }
    for (const auto& prev : out) {
      if (prev.key == c.key)
        throw std::runtime_error("duplicate base OAK camera key '" + c.key +
                                 "' in " + path);
    }

    if (const YAML::Node rig = cam["rig"]) {
      if (const YAML::Node t = rig["translation"]) {
        if (t.size() != 3)
          throw std::runtime_error("rig.translation must have 3 elements for " +
                                   c.key);
        c.translation = {t[0].as<double>(), t[1].as<double>(), t[2].as<double>()};
      }
      if (const YAML::Node r = rig["rotation"]) {
        c.roll_deg = r["roll"] ? r["roll"].as<double>() : 0.0;
        c.pitch_deg = r["pitch"] ? r["pitch"].as<double>() : 0.0;
        c.yaw_deg = r["yaw"] ? r["yaw"].as<double>() : 0.0;
      }
    }

    if (const YAML::Node ext = cam["stereo_extrinsic"]) {
      const YAML::Node r = ext["rotation"];
      const YAML::Node t = ext["translation"];
      if (!r || r.size() != 4)
        throw std::runtime_error(
            "stereo_extrinsic.rotation must have 4 elements for " + c.key);
      if (!t || t.size() != 3)
        throw std::runtime_error(
            "stereo_extrinsic.translation must have 3 elements for " + c.key);
      // Config order: [qx, qy, qz, qw]
      c.right_from_left_rot = {r[0].as<double>(), r[1].as<double>(),
                               r[2].as<double>(), r[3].as<double>()};
      c.right_from_left_trans = {t[0].as<double>(), t[1].as<double>(),
                                 t[2].as<double>()};
      c.has_stereo_extrinsic = true;
    }

    out.push_back(std::move(c));
  }

  if (out.empty()) {
    throw std::runtime_error("no base OAK camera in " + path);
  }
  return out;
}

}  // namespace rena_cuvslam
