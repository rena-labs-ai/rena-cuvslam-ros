// Shared /etc/rena/config.yaml parsing for the RGBD and Stereo trackers.
#include "rena_cuvslam_ros/tracker_common.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace rena_cuvslam {
namespace {
constexpr char kConfigPath[] = "/etc/rena/config.yaml";
}  // namespace

std::vector<OakCameraConfig> load_base_oak_cameras() {
  YAML::Node root = YAML::LoadFile(kConfigPath);
  const YAML::Node base = root["base"];
  if (!base || !base["cameras"]) {
    throw std::runtime_error("no base.cameras in " + std::string(kConfigPath));
  }

  std::vector<OakCameraConfig> out;
  for (const auto& cam : base["cameras"]) {
    if (!cam["type"] || cam["type"].as<std::string>() != "oak") continue;

    OakCameraConfig c;
    c.key = cam["key"] ? cam["key"].as<std::string>() : "";
    c.serial_no = cam["serial_no"] ? cam["serial_no"].as<std::string>() : "";

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
    throw std::runtime_error("no base OAK camera in " + std::string(kConfigPath));
  }
  return out;
}

}  // namespace rena_cuvslam
