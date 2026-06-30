// Frame / rig math for the cuVSLAM RGBD tracker.
//
// Ported 1:1 from the Python rena_cuvslam_ros.tracker helpers so the C++ node
// produces bit-identical extrinsics and pose output:
//   - quaternion <-> rotation matrix
//   - cuVSLAM (OpenCV optical) -> canonical ROS frame conversion
//   - robot-body camera pose -> cuVSLAM optical rig_from_camera
//
// Pure header (no ROS / no cuVSLAM dependency) so it can be unit-checked
// in isolation. Quaternions are (x, y, z, w), matrices are row-major.
#pragma once

#include <array>
#include <cmath>

namespace rena_cuvslam {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;            // (x, y, z, w)
using Mat3 = std::array<std::array<double, 3>, 3>;  // row-major

// inverse of cuvslam_pose_canonical in isaac_ros_visual_slam
// (cuvslam_ros_conversion.hpp): v_ros = T @ v_cuv.
inline constexpr Mat3 kTCuvToRos = {{
    {{0.0, 0.0, 1.0}},
    {{-1.0, 0.0, 0.0}},
    {{0.0, -1.0, 0.0}},
}};

// Rotation that maps robot body axes (x-fwd, y-left, z-up) into cuVSLAM
// optical axes (x-right, y-down, z-fwd). Matches Python _R_OPT_FROM_ROBOT.
inline constexpr Mat3 kROptFromRobot = {{
    {{0.0, -1.0, 0.0}},
    {{0.0, 0.0, -1.0}},
    {{1.0, 0.0, 0.0}},
}};

inline Mat3 mat3_mul(const Mat3& a, const Mat3& b) {
  Mat3 out{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += a[i][k] * b[k][j];
      out[i][j] = s;
    }
  return out;
}

inline Mat3 mat3_transpose(const Mat3& a) {
  Mat3 out{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i][j] = a[j][i];
  return out;
}

inline Vec3 mat3_apply(const Mat3& a, const Vec3& v) {
  return {
      a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
      a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
      a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
  };
}

// Column-vector convention: v' = R @ v.
inline Mat3 quat_to_rotmat(const Quat& q) {
  const double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
  const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
  const double wx = qw * qx, wy = qw * qy, wz = qw * qz;
  return {{
      {{1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)}},
      {{2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)}},
      {{2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)}},
  }};
}

// Rotation matrix (row-major) to quaternion (x, y, z, w).
inline Quat rotmat_to_quat(const Mat3& m) {
  const double tr = m[0][0] + m[1][1] + m[2][2];
  double qx, qy, qz, qw;
  if (tr > 0.0) {
    const double s = 0.5 / std::sqrt(tr + 1.0);
    qw = 0.25 / s;
    qx = (m[2][1] - m[1][2]) * s;
    qy = (m[0][2] - m[2][0]) * s;
    qz = (m[1][0] - m[0][1]) * s;
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const double s = 2.0 * std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
    qx = 0.25 * s;
    qy = (m[0][1] + m[1][0]) / s;
    qz = (m[0][2] + m[2][0]) / s;
    qw = (m[2][1] - m[1][2]) / s;
  } else if (m[1][1] > m[2][2]) {
    const double s = 2.0 * std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
    qx = (m[0][1] + m[1][0]) / s;
    qy = 0.25 * s;
    qz = (m[1][2] + m[2][1]) / s;
    qw = (m[0][2] - m[2][0]) / s;
  } else {
    const double s = 2.0 * std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
    qx = (m[0][2] + m[2][0]) / s;
    qy = (m[1][2] + m[2][1]) / s;
    qz = 0.25 * s;
    qw = (m[1][0] - m[0][1]) / s;
  }
  return {qx, qy, qz, qw};
}

// R_ros = T @ R_cuv @ T^T with the Isaac canonical T (cuVSLAM -> ROS).
inline Quat cuv_to_ros_rotation(const Quat& q_cuv) {
  const Mat3 r_cuv = quat_to_rotmat(q_cuv);
  const Mat3 r_ros =
      mat3_mul(kTCuvToRos, mat3_mul(r_cuv, mat3_transpose(kTCuvToRos)));
  return rotmat_to_quat(r_ros);
}

// cuVSLAM pose (optical frame) -> canonical ROS robot frame.
// translation (z, -x, -y); rotation R_ros = T R_cuv T^T. Matches Python
// Pose.to_robot_frame() / isaac_ros_visual_slam canonical_pose_cuvslam.
struct RosPose {
  Vec3 translation;
  Quat rotation;  // (x, y, z, w)
};

inline RosPose to_robot_frame(const Quat& rot_cuv, const Vec3& t_cuv) {
  return RosPose{
      {t_cuv[2], -t_cuv[0], -t_cuv[1]},
      cuv_to_ros_rotation(rot_cuv),
  };
}

// Extrinsic XYZ Euler (radians) -> rotation matrix, matching
// scipy Rotation.from_euler("xyz", ...): R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
inline Mat3 euler_xyz_to_rotmat(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll), sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  const Mat3 rx = {{{{1, 0, 0}}, {{0, cr, -sr}}, {{0, sr, cr}}}};
  const Mat3 ry = {{{{cp, 0, sp}}, {{0, 1, 0}}, {{-sp, 0, cp}}}};
  const Mat3 rz = {{{{cy, -sy, 0}}, {{sy, cy, 0}}, {{0, 0, 1}}}};
  return mat3_mul(rz, mat3_mul(ry, rx));
}

// Camera pose given in robot body frame -> rig_from_camera in cuVSLAM's
// optical rig convention. Matches Python _rig_from_camera_from_robot_pose:
//   R_rig = C @ R_robot @ C^T ,  t_rig = C @ t_robot   (C = kROptFromRobot)
// roll/pitch/yaw are in DEGREES (as stored in /etc/rena/config.yaml).
struct RigFromCamera {
  Quat rotation;  // (x, y, z, w)
  Vec3 translation;
};

inline RigFromCamera rig_from_camera_from_robot_pose(double roll_deg,
                                                     double pitch_deg,
                                                     double yaw_deg,
                                                     const Vec3& t_robot) {
  constexpr double kDeg2Rad = M_PI / 180.0;
  const Mat3 r_robot = euler_xyz_to_rotmat(
      roll_deg * kDeg2Rad, pitch_deg * kDeg2Rad, yaw_deg * kDeg2Rad);
  const Mat3 r_rig =
      mat3_mul(kROptFromRobot, mat3_mul(r_robot, mat3_transpose(kROptFromRobot)));
  const Vec3 t_rig = mat3_apply(kROptFromRobot, t_robot);
  return RigFromCamera{rotmat_to_quat(r_rig), t_rig};
}

// Yaw from a (x, y, z, w) quaternion — used for the planarized TF.
inline double quat_to_yaw(const Quat& q) {
  const double siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1]);
  const double cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
  return std::atan2(siny_cosp, cosy_cosp);
}

}  // namespace rena_cuvslam
