"""OAK stereo / RGBD cuVSLAM trackers and the shared tracking infrastructure
they depend on.

Self-contained (no cuvslam_examples dependency): bundles the cuVSLAM->ROS
pose/landmark frame conversions, the TrackingResult dataclass, the BaseTracker
interface, and the ROS CameraInfo / compressed-depth helpers the OAK trackers
use. Pure tracking logic — ROS node wiring lives in node.py.
"""

import math
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

import cuvslam as vslam


# ---------- cuVSLAM (OpenCV) -> canonical ROS frame conversions ----------
# inverse of cuvslam_pose_canonical in
# isaac_ros_visual_slam/impl/cuvslam_ros_conversion.hpp (v_ros = T @ v_cuv).
_T_CUV_TO_ROS = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


def _mat3_mul(
    a: Tuple[Tuple[float, ...], ...], b: Tuple[Tuple[float, ...], ...]
) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _mat3_transpose(a: Tuple[Tuple[float, ...], ...]) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def _quat_to_rotmat(
    qx: float, qy: float, qz: float, qw: float
) -> Tuple[Tuple[float, float, float], ...]:
    """Column-vector convention: v' = R @ v."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _rotmat_to_quat(
    r: Tuple[Tuple[float, ...], ...],
) -> Tuple[float, float, float, float]:
    """Rotation matrix (row-major rows) to quaternion (qx, qy, qz, qw)."""
    m = r
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (m[2][1] - m[1][2]) * s
        qy = (m[0][2] - m[2][0]) * s
        qz = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
        qw = (m[2][1] - m[1][2]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
        qw = (m[0][2] - m[2][0]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
        qw = (m[1][0] - m[0][1]) / s
    return qx, qy, qz, qw


def _cuv_to_ros_rotation(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float, float]:
    """R_ros = T @ R_cuv @ T.T with Isaac canonical T (cuVSLAM -> ROS)."""
    t = _T_CUV_TO_ROS
    r_cuv = _quat_to_rotmat(qx, qy, qz, qw)
    r_ros = _mat3_mul(t, _mat3_mul(r_cuv, _mat3_transpose(t)))
    return _rotmat_to_quat(r_ros)


class Pose(vslam.Pose):
    """Pose in camera frame."""

    def __init__(self, pose: vslam.Pose) -> None:
        super().__init__(pose.rotation, pose.translation)

    def to_robot_frame(self) -> "Pose":
        """Convert pose from cuVSLAM (OpenCV) camera frame to canonical ROS frame.

        Matches isaac_ros_visual_slam ``canonical_pose_cuvslam``: translation
        ``(z, -x, -y)`` and rotation ``R_ros = T R_cuv T^T``.
        """
        qx, qy, qz, qw = self.rotation
        tx, ty, tz = self.translation[0], self.translation[1], self.translation[2]
        robot_rot = _cuv_to_ros_rotation(qx, qy, qz, qw)
        robot_pose = vslam.Pose(robot_rot, [tz, -tx, -ty])
        return Pose(robot_pose)


class Landmark:
    """Landmark in camera frame."""

    def __init__(self, id: int, coords: List[float]) -> None:
        self.id = id
        self.coords = coords

    def to_robot_frame(self) -> "Landmark":
        """Convert the landmark to robot frame."""
        robot = object.__new__(Landmark)
        robot.id = self.id
        robot.coords = [self.coords[2], -self.coords[0], -self.coords[1]]
        return robot


# ---------- TrackingResult ----------
@dataclass
class TrackingResult:
    """Single tracking frame output from a pipeline."""

    timestamp: int
    vo_pose: Pose
    slam_pose: Pose
    images: Tuple[np.ndarray, ...]
    landmarks: List = field(default_factory=list)
    synced_odom: Optional[Any] = None  # Ground truth at capture time (e.g. nav_msgs/Odometry)


# ---------- BaseTracker interface ----------
class BaseTracker(ABC):
    """Abstract interface for cuVSLAM tracking strategies."""

    @property
    def num_viz_cameras(self) -> int:
        return 1

    @abstractmethod
    def setup_camera_parameters(self) -> dict: ...

    @abstractmethod
    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig: ...

    @abstractmethod
    def create_rig(self, camera_params: dict) -> vslam.Rig: ...

    @abstractmethod
    def create_slam_config(self) -> vslam.Tracker.SlamConfig: ...

    @abstractmethod
    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None: ...

    @abstractmethod
    def stop_streaming(self) -> None: ...

    def get_viz_image_indices(self) -> List[int]:
        """Image indices from TrackingResult.images to visualize."""
        return [0]

    def get_viz_observation_indices(self) -> List[int]:
        """Camera indices passed to get_last_observations for each viz camera."""
        return [0]


# ---------- ROS CameraInfo / compressed-depth helpers ----------
class _CameraInfoCollector:
    """Accumulates CameraInfo messages keyed by topic identifier."""

    def __init__(self, keys) -> None:
        self.received = {k: None for k in keys}

    def on_info(self, key: str, msg) -> None:
        if self.received[key] is None:
            self.received[key] = msg

    def has_all(self) -> bool:
        return all(v is not None for v in self.received.values())


def _spin_ros_node(node, tracker) -> None:
    import rclpy

    while tracker._running and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)


# ---------- Local helpers for raw-image OAK stereo (no realsense dep) ----------

def _oak_image_to_raw_camera_info_topic(image_topic: str) -> str:
    """OAK raw convention: /base/front/left/image_raw -> /base/front/left/camera_info"""
    parent = image_topic.rsplit("/", 1)[0]
    return parent + "/camera_info"


def _make_oak_raw_camera(k, d, width: int, height: int, rig_from_camera_4x4) -> vslam.Camera:
    """Build a cuvslam.Camera from raw K, D (ROS rational_polynomial), image size,
    and the 4x4 pose of this camera in the rig frame.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    cam = vslam.Camera()
    # K is a flat 9-list from CameraInfo.k
    cam.focal = (k[0], k[4])
    cam.principal = (k[2], k[5])
    cam.size = (width, height)
    # cuvslam Polynomial = OpenCV rational polynomial, exactly 8 coeffs:
    # [k1, k2, p1, p2, k3, k4, k5, k6]. ROS rational_polynomial D has 14
    # elements; the first 8 are this set (indices 8..13 are thin-prism /
    # tilted terms that cuvslam's Polynomial doesn't model).
    d8 = [float(d[i]) if i < len(d) else 0.0 for i in range(8)]
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial, d8)

    m = np.asarray(rig_from_camera_4x4, dtype=np.float64)
    cam.rig_from_camera = vslam.Pose(
        rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
        translation=m[:3, 3],
    )
    return cam

SLOP_SEC = 0.033
QUEUE_SIZE = 2


# Rotation that maps robot body axes (x-fwd, y-left, z-up) into cuVSLAM
# optical axes (x-right, y-down, z-fwd). Rows are optical basis vectors
# expressed in robot coordinates; equivalently,
#   R_OPT_FROM_ROBOT @ [x_fwd, y_left, z_up] = [-y, -z, x] = [right, down, fwd].
_R_OPT_FROM_ROBOT = [
    [0, -1, 0],
    [0,  0, -1],
    [1,  0, 0],
]


def _rig_from_camera_from_robot_pose(rotation_cfg: dict, translation_cfg):
    """Convert a camera pose given in robot body frame to the equivalent
    rig_from_camera 4x4 in cuVSLAM's optical rig convention.

    The rig frame is kept in optical axes (x-right, y-down, z-fwd) so that the
    default ``Pose.to_robot_frame()`` (cuVSLAM -> ROS) still applies to the
    tracker output. Config values are intuitive robot-frame angles/offsets,
    which we rebase here:

        R_rig = R_OPT_FROM_ROBOT @ R_robot @ R_OPT_FROM_ROBOT.T
        t_rig = R_OPT_FROM_ROBOT @ t_robot
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    rpy = [
        float((rotation_cfg or {}).get("roll", 0.0)),
        float((rotation_cfg or {}).get("pitch", 0.0)),
        float((rotation_cfg or {}).get("yaw", 0.0)),
    ]
    t_robot = np.asarray(translation_cfg or [0.0, 0.0, 0.0], dtype=np.float64)
    if t_robot.shape != (3,):
        raise ValueError(
            f"rig.translation must be a 3-element list, got {translation_cfg!r}"
        )

    R_robot = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    C = np.asarray(_R_OPT_FROM_ROBOT, dtype=np.float64)
    R_rig = C @ R_robot @ C.T
    t_rig = C @ t_robot

    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R_rig
    m[:3, 3] = t_rig
    return m


def _oak_image_to_compressed_topic(image_topic: str) -> str:
    """OAK images are recorded as compressed: /base/front/left/image_raw/compressed"""
    return image_topic + "/compressed"


def _decode_oak_compressed_image(msg) -> "np.ndarray | None":
    import cv2
    import numpy as np

    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return np.ascontiguousarray(img) if img is not None else None


def _decode_oak_raw_color(msg) -> "np.ndarray | None":
    """Decode a sensor_msgs/Image color frame from the depthai driver into a
    grayscale numpy array (cuvslam features are luma-only)."""
    import cv2
    import numpy as np

    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("mono8", "8uc1"):
        return np.ascontiguousarray(buf.reshape(h, w))
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, w, 3)
        code = cv2.COLOR_RGB2GRAY if enc == "rgb8" else cv2.COLOR_BGR2GRAY
        return np.ascontiguousarray(cv2.cvtColor(img, code))
    return None


def _decode_oak_raw_depth(msg) -> "np.ndarray | None":
    """Decode a sensor_msgs/Image uint16 depth frame (16UC1 / mono16)."""
    import numpy as np

    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    if enc not in ("16uc1", "mono16"):
        return None
    return np.ascontiguousarray(
        np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w)
    )


def _load_rena_oak_cameras():
    """Read the single robot config at /etc/rena/config.yaml (written by
    rena-commission) and return a flat list of its OAK cameras:

        [{"serial_no": str, "key": str, "robot_part": "base"|"arm",
          "image_mode": "raw"|"rect"|"alternate", "rect_calib": dict|None}, ...]
    """
    import yaml

    config_path = "/etc/rena/config.yaml"

    with open(config_path) as f:
        robot_cfg = yaml.safe_load(f) or {}

    out = []
    for part in ("base", "arm"):
        part_cfg = robot_cfg.get(part) or {}
        for cam in part_cfg.get("cameras", []) or []:
            if cam.get("type") != "oak":
                continue
            out.append({
                "serial_no": cam.get("serial_no"),
                "key": cam.get("key"),
                "robot_part": part,
                "image_mode": cam.get("image_mode", "raw"),
                "rect_calib": cam.get("rect_calib"),
                "rig": cam.get("rig"),
                "stereo_extrinsic": cam.get("stereo_extrinsic"),
            })
    return out


def _oak_image_topic_suffix(image_mode: str) -> str:
    """Map config image_mode to ROS topic segment (image_raw vs image_rect)."""
    if image_mode in ("raw", "alternate"):
        return "raw"
    if image_mode == "rect":
        return "rect"
    raise ValueError(
        f"unsupported OAK image_mode {image_mode!r}; "
        "expected 'raw', 'rect', or 'alternate'"
    )


def _oak_topics_for_entry(entry: dict) -> tuple[str, str]:
    """Return (left_topic, right_topic) for a config.yaml OAK entry, using its
    image_mode to pick the image_raw / image_rect suffix."""
    suffix = f"image_{_oak_image_topic_suffix(entry['image_mode'])}"
    ns = f"/{entry['robot_part']}/{entry['key']}"
    # return f"{ns}/dot_off/left/{suffix}", f"{ns}/dot_off/right/{suffix}"
    return f"{ns}/left/{suffix}", f"{ns}/right/{suffix}"


class RosOakStereoTracker(BaseTracker):
    """Unified OAK stereo tracker — single or multi-camera.

    Reads every OAK entry from rena_bringup/config/config.yaml and auto-picks
    cuVSLAM stereo (1 pair) vs multicam (N pairs). All inter-unit extrinsics
    live in config.yaml under each entry's ``rig:`` block (translation +
    rotation in robot body frame, x-fwd/y-left/z-up); the tracker rebases that
    into cuVSLAM's optical rig convention before calling ``track()``. The
    within-unit L/R baseline always comes from ``rect_calib.baseline_m``.

    Images are fed to tracker.track() in order:
        [cam0_left, cam0_right, cam1_left, cam1_right, ...]
    """

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        oaks = _load_rena_oak_cameras()
        if not oaks:
            raise RuntimeError(
                "RosOakStereoTracker: no OAK cameras found in rena_bringup config.yaml."
            )

        # cuVSLAM's rectified_stereo_camera is a rig-wide flag; all OAKs must
        # agree on raw vs rect streams (raw and alternate both use image_raw).
        modes = {o["image_mode"] for o in oaks}
        try:
            kinds = {_oak_image_topic_suffix(m) for m in modes}
        except ValueError as e:
            raise RuntimeError(f"RosOakStereoTracker: {e}") from e
        if len(kinds) > 1:
            raise RuntimeError(
                f"RosOakStereoTracker: mixed image_mode across OAKs ({sorted(modes)}). "
                "cuVSLAM requires all cameras to share one rectified_stereo_camera "
                "setting — make every OAK's image_mode match."
            )
        self._rect_cam_info = kinds == {"rect"}
        self._entries = oaks
        for e in self._entries:
            left_topic, right_topic = _oak_topics_for_entry(e)
            e["left_topic"] = left_topic
            e["right_topic"] = right_topic

        mode_label = "RECT" if self._rect_cam_info else "RAW"
        print(f"[ros_oak_stereo] mode={mode_label}  N={len(self._entries)}")
        for i, e in enumerate(self._entries):
            print(
                f"  cam{i}: serial={e['serial_no']} {e['robot_part']}/{e['key']}  "
                f"topics: {e['left_topic']} | {e['right_topic']}"
            )

        self._running = False

    @property
    def num_viz_cameras(self) -> int:
        return len(self._entries)

    def get_viz_image_indices(self) -> List[int]:
        return list(range(0, len(self._entries) * 2, 2))

    def get_viz_observation_indices(self) -> List[int]:
        return list(range(0, len(self._entries) * 2, 2))

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        per_entry: List[Dict] = []

        if self._rect_cam_info:
            for entry in self._entries:
                calib = entry["rect_calib"]
                if calib is None:
                    raise RuntimeError(
                        f"OAK serial {entry['serial_no']} in config.yaml is "
                        f"missing its rect_calib block (required for rect mode)."
                    )
                per_entry.append({"entry": entry, "calib": calib})
            return {"entries": per_entry}

        # Raw mode: batch-collect every CameraInfo pair. Baseline comes from
        # rect_calib.baseline_m (not /tf) — the TF baseline can drift; rect_calib
        # is the factory stereo calibration and keeps the scale correct.
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        keys: List[str] = []
        for i in range(len(self._entries)):
            keys += [f"cam{i}_left", f"cam{i}_right"]

        print(
            f"[ros_oak_stereo] Waiting for raw CameraInfo on {len(keys)} topics ..."
        )
        collector = _CameraInfoCollector(keys)
        node = Node("ros_oak_stereo_camera_info")
        for i, entry in enumerate(self._entries):
            left_info = _oak_image_to_raw_camera_info_topic(entry["left_topic"])
            right_info = _oak_image_to_raw_camera_info_topic(entry["right_topic"])
            node.create_subscription(
                CameraInfo, left_info, partial(collector.on_info, f"cam{i}_left"), 10
            )
            node.create_subscription(
                CameraInfo, right_info, partial(collector.on_info, f"cam{i}_right"), 10
            )

        deadline = time.time() + 5.0
        while not collector.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()

        missing = [k for k, v in collector.received.items() if v is None]
        if missing:
            raise TimeoutError(
                f"Did not receive CameraInfo for {missing} within 5 s"
            )
        print("[ros_oak_stereo] raw CameraInfo received for all cameras")

        for i, entry in enumerate(self._entries):
            if entry["rect_calib"] is None:
                raise RuntimeError(
                    f"OAK serial {entry['serial_no']} in config.yaml is missing "
                    "its rect_calib block (needed for baseline_m in raw mode too)."
                )
            per_entry.append({
                "entry": entry,
                "left_msg": collector.received[f"cam{i}_left"],
                "right_msg": collector.received[f"cam{i}_right"],
            })

        return {"entries": per_entry}

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        return vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            rectified_stereo_camera=self._rect_cam_info,
            use_denoising=False,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        import numpy as np
        from scipy.spatial.transform import Rotation

        def _mk_rect(calib: dict, rig_from_cam: "np.ndarray") -> vslam.Camera:
            cam = vslam.Camera()
            cam.focal = (calib["fx"], calib["fy"])
            cam.principal = (calib["cx"], calib["cy"])
            cam.size = (calib["width"], calib["height"])
            cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
            m = np.asarray(rig_from_cam, dtype=np.float64)
            cam.rig_from_camera = vslam.Pose(
                rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                translation=m[:3, 3],
            )
            return cam

        cameras: List[vslam.Camera] = []
        for p in camera_params["entries"]:
            entry = p["entry"]
            rig_cfg = entry.get("rig") or {}
            rig_from_left = _rig_from_camera_from_robot_pose(
                rig_cfg.get("rotation"),
                rig_cfg.get("translation"),
            )
            print(
                f"[ros_oak_stereo] {entry['robot_part']}/{entry['key']}: "
                f"rig_from_left t={rig_from_left[:3, 3].tolist()}"
            )

            # rig_from_right = rig_from_left @ left_from_right.
            #
            # In rect mode: rectified stereo guarantees parallel optical axes,
            # so left_from_right is just a +x baseline shim (rect_calib.baseline_m).
            #
            # In raw mode: cuVSLAM does its own triangulation from the raw
            # frames and needs the *true* L<->R rigid transform — small (~1deg)
            # rotation + (ty, tz) offsets are physically present and ignoring
            # them gives wrong-depth landmarks and short trajectories. Use
            # entry['stereo_extrinsic'] (right_from_left, factory) when given.
            baseline_m = entry["rect_calib"]["baseline_m"]
            stereo_ext = entry.get("stereo_extrinsic")
            if self._rect_cam_info or stereo_ext is None:
                if not self._rect_cam_info:
                    print(
                        f"[ros_oak_stereo] WARN {entry['robot_part']}/"
                        f"{entry['key']}: raw mode without stereo_extrinsic; "
                        "falling back to +x baseline shim — tracking will drift."
                    )
                left_from_right = np.eye(4)
                left_from_right[0, 3] = baseline_m
            else:
                ext_t = np.asarray(stereo_ext["translation"], dtype=np.float64)
                ext_q = np.asarray(stereo_ext["rotation"], dtype=np.float64)
                if ext_t.shape != (3,) or ext_q.shape != (4,):
                    raise ValueError(
                        f"stereo_extrinsic for {entry['serial_no']} must have "
                        "translation (3) and rotation (4 = qx,qy,qz,qw)"
                    )
                right_from_left = np.eye(4)
                right_from_left[:3, :3] = Rotation.from_quat(ext_q).as_matrix()
                right_from_left[:3, 3] = ext_t
                left_from_right = np.linalg.inv(right_from_left)
            rig_from_right = rig_from_left @ left_from_right

            if self._rect_cam_info:
                calib = p["calib"]
                cam_left = _mk_rect(calib, rig_from_left)
                cam_right = _mk_rect(calib, rig_from_right)
                k_left = [calib["fx"], 0, calib["cx"],
                          0, calib["fy"], calib["cy"], 0, 0, 1]
                k_right = list(k_left)
                d_left = [0.0] * 8
                d_right = [0.0] * 8
            else:
                left_msg = p["left_msg"]
                right_msg = p["right_msg"]
                cam_left = _make_oak_raw_camera(
                    k=left_msg.k, d=left_msg.d,
                    width=left_msg.width, height=left_msg.height,
                    rig_from_camera_4x4=rig_from_left,
                )
                cam_right = _make_oak_raw_camera(
                    k=right_msg.k, d=right_msg.d,
                    width=right_msg.width, height=right_msg.height,
                    rig_from_camera_4x4=rig_from_right,
                )
                k_left = list(left_msg.k)
                k_right = list(right_msg.k)
                d_left = [float(left_msg.d[i]) if i < len(left_msg.d) else 0.0
                          for i in range(8)]
                d_right = [float(right_msg.d[i]) if i < len(right_msg.d) else 0.0
                           for i in range(8)]

            cameras.append(cam_left)
            cameras.append(cam_right)

            np.set_printoptions(precision=6, suppress=True)
            label = f"{entry['robot_part']}/{entry['key']}"
            mode = "RECT" if self._rect_cam_info else "RAW"
            for side, cam, k, d in (("LEFT (CAM_B)", cam_left, k_left, d_left),
                                    ("RIGHT (CAM_C)", cam_right, k_right, d_right)):
                K = np.asarray(k, dtype=np.float64).reshape(3, 3)
                D = np.asarray(d, dtype=np.float64)
                rfc_t = np.asarray(cam.rig_from_camera.translation)
                rfc_q = np.asarray(cam.rig_from_camera.rotation)
                print()
                print(f"--- [ros_oak_stereo] {label} {side}  mode={mode} ---")
                print(f"size = {cam.size}")
                print(f"focal = {cam.focal}")
                print(f"principal = {cam.principal}")
                print(f"K =\n{K}")
                print(f"D (8) = {D}")
                print(f"distortion model = {cam.distortion.model}")
                print(f"rig_from_camera.translation [m] = {rfc_t}")
                print(f"rig_from_camera.rotation (qx,qy,qz,qw) = {rfc_q}")

            rig_t_left = np.asarray(cam_left.rig_from_camera.translation)
            rig_t_right = np.asarray(cam_right.rig_from_camera.translation)
            rig_baseline_m = float(np.linalg.norm(rig_t_right - rig_t_left))
            print()
            print(f"[ros_oak_stereo] {label} baseline (rect_calib.baseline_m): "
                  f"{baseline_m:.6f} m")
            print(f"[ros_oak_stereo] {label} baseline (||rig_t_R - rig_t_L||):  "
                  f"{rig_baseline_m:.6f} m")
            print()

        # for cam in cameras:
        #     cam.border_bottom = cam.size[1] // 4

        rig = vslam.Rig()
        rig.cameras = cameras
        return rig

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import message_filters
        from sensor_msgs.msg import CompressedImage, Image
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._running = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        frames_fed = [0]
        last_log_time = [time.monotonic()]
        last_ts = [0]

        def on_frames(*msgs):
            ts = (
                msgs[0].header.stamp.sec * 1_000_000_000
                + msgs[0].header.stamp.nanosec
            )
            if ts <= last_ts[0]:
                return
            last_ts[0] = ts

            t_decode = time.monotonic()
            images = [_decode_oak_compressed_image(m) for m in msgs]
            decode_ms = (time.monotonic() - t_decode) * 1000

            if any(img is None for img in images):
                print("[ros_oak_stereo] Warning: decode failed", flush=True)
                return

            t0 = time.monotonic()
            vo_pose_estimate, slam_pose = tracker.track(ts, images)
            track_ms = (time.monotonic() - t0) * 1000

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                print("[ros_oak_stereo] Warning: track failed", flush=True)
                return

            frames_fed[0] += 1
            now = time.monotonic()
            if self._debug and now - last_log_time[0] >= 1.0:
                print(
                    f"[ros_oak_stereo] fed={frames_fed[0]}/s  "
                    f"decode={decode_ms:.1f}ms  track={track_ms:.1f}ms",
                    flush=True,
                )
                frames_fed[0] = 0
                last_log_time[0] = now

            landmarks = [
                Landmark(lm.id, lm.coords) for lm in tracker.get_last_landmarks()
            ]
            output_queue.put(
                TrackingResult(
                    ts,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose),
                    images,
                    landmarks,
                )
            )

        self._node = Node("ros_oak_stereo_frames")
        subscribers = []
        topics_log: List[str] = []
        for entry in self._entries:
            for topic in (entry["left_topic"], entry["right_topic"]):
                compressed = _oak_image_to_compressed_topic(topic)
                subscribers.append(
                    message_filters.Subscriber(
                        self._node, CompressedImage, compressed, qos_profile=qos
                    )
                )
                topics_log.append(compressed)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=100, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_frames)

        print(
            f"[ros_oak_stereo] Subscribed (ApproximateTimeSynchronizer "
            f"slop={SLOP_SEC*1000:.0f}ms): " + ", ".join(topics_log),
            flush=True,
        )

        self._spin_thread = threading.Thread(
            target=_spin_ros_node, args=(self._node, self), daemon=True
        )
        self._spin_thread.start()

    def stop_streaming(self) -> None:
        self._running = False
        if hasattr(self, "_spin_thread"):
            self._spin_thread.join(timeout=2.0)
        if hasattr(self, "_node"):
            self._node.destroy_node()


class RosOakRGBDTracker(BaseTracker):
    """OAK RGBD tracker (raw RGB + raw depth) — single or multi-camera.

    The number of base OAK cameras in /etc/rena/config.yaml selects the mode:
    one camera keeps the classic single-RGBD odometry; two or more enable
    cuVSLAM multi-depth ICP (cuVSLAM#3), fusing every camera's depth track.

    Each base OAK contributes one RGB-D stream; color and depth are both
    consumed RAW (sensor_msgs/Image):
      color: /base/<key>/rgb/image_raw
      depth: /base/<key>/stereo/image_raw

    Per-camera rig extrinsics come from each entry's ``rig:`` block (robot body
    frame, x-fwd/y-left/z-up), rebased into cuVSLAM's optical rig convention.
    Frames are fed to track() as images=[color0, color1, ...],
    depths=[depth0, depth1, ...]. Depth is uint16 millimeters by default
    (depth_scale=0.001).
    """

    def __init__(self, depth_scale: float = 0.001, debug: bool = False) -> None:
        self._debug = debug

        oaks = [o for o in _load_rena_oak_cameras() if o["robot_part"] == "base"]
        if not oaks:
            raise RuntimeError(
                "RosOakRGBDTracker: no base OAK camera in /etc/rena/config.yaml."
            )
        self._entries = oaks
        self._depth_scale = depth_scale
        self._running = False

        for e in self._entries:
            ns = f"/{e['robot_part']}/{e['key']}"
            e["color_topic"] = f"{ns}/rgb/image_raw"
            e["depth_topic"] = f"{ns}/stereo/image_raw"

        n = len(self._entries)
        print(f"[ros_oak_rgbd] mode={'MULTI' if n > 1 else 'SINGLE'}  N={n}")
        for i, e in enumerate(self._entries):
            print(
                f"  cam{i}: serial={e['serial_no']} {e['robot_part']}/{e['key']}\n"
                f"    color: {e['color_topic']}\n"
                f"    depth: {e['depth_topic']}"
            )

    @property
    def num_viz_cameras(self) -> int:
        return len(self._entries)

    def get_viz_image_indices(self) -> List[int]:
        return list(range(len(self._entries)))

    def get_viz_observation_indices(self) -> List[int]:
        return list(range(len(self._entries)))

    def setup_camera_parameters(self) -> Dict[str, Dict]:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo

        keys = [f"cam{i}_color" for i in range(len(self._entries))]
        collector = _CameraInfoCollector(keys)
        node = Node("ros_oak_rgbd_camera_info")
        for i, entry in enumerate(self._entries):
            info_topic = _oak_image_to_raw_camera_info_topic(entry["color_topic"])
            print(f"[ros_oak_rgbd] Waiting for CameraInfo on {info_topic} ...")
            node.create_subscription(
                CameraInfo, info_topic, partial(collector.on_info, f"cam{i}_color"), 10
            )

        deadline = time.time() + 30.0
        while not collector.has_all() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()

        missing = [k for k, v in collector.received.items() if v is None]
        if missing:
            raise TimeoutError(
                f"Did not receive CameraInfo for {missing} within 30 s"
            )
        print("[ros_oak_rgbd] CameraInfo received for all cameras")
        return {
            "color_msgs": [
                collector.received[f"cam{i}_color"]
                for i in range(len(self._entries))
            ]
        }

    def create_odometry_config(self) -> vslam.Tracker.OdometryConfig:
        rgbd_settings = vslam.Tracker.OdometryRGBDSettings()
        rgbd_settings.enable_depth_stereo_tracking = False
        scale_factor = 1.0 / self._depth_scale

        if len(self._entries) == 1:
            rgbd_settings.depth_scale_factor = scale_factor
            rgbd_settings.depth_camera_id = 0
        else:
            # Multi camera: one depth source per rig camera (cuVSLAM#3).
            try:
                DepthCam = vslam.Tracker.OdometryRGBDDepthCameraSettings
            except AttributeError as e:
                raise RuntimeError(
                    "Multi-camera RGBD needs the rena cuVSLAM build with "
                    "multi-depth ICP (cuVSLAM#3, feat/add-multi-depth-icp); the "
                    "installed cuvslam lacks OdometryRGBDDepthCameraSettings."
                ) from e
            rgbd_settings.depth_cameras = [
                DepthCam(camera_id=i, depth_scale_factor=scale_factor)
                for i in range(len(self._entries))
            ]

        return vslam.Tracker.OdometryConfig(
            async_sba=True,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            odometry_mode=vslam.Tracker.OdometryMode.RGBD,
            rgbd_settings=rgbd_settings,
            rectified_stereo_camera=False,
        )

    def create_slam_config(self) -> vslam.Tracker.SlamConfig:
        return vslam.Tracker.SlamConfig(sync_mode=False, planar_constraints=True)

    def create_rig(self, camera_params: dict) -> vslam.Rig:
        import numpy as np

        np.set_printoptions(precision=6, suppress=True)
        cameras: List[vslam.Camera] = []
        for entry, msg in zip(self._entries, camera_params["color_msgs"]):
            rig_cfg = entry.get("rig") or {}
            rig_from_color = _rig_from_camera_from_robot_pose(
                rig_cfg.get("rotation"),
                rig_cfg.get("translation"),
            )
            cam = _make_oak_raw_camera(
                k=msg.k, d=msg.d,
                width=msg.width, height=msg.height,
                rig_from_camera_4x4=rig_from_color,
            )
            cameras.append(cam)

            # Calibration dump — same format as RosOakStereoTracker.create_rig.
            label = f"{entry['robot_part']}/{entry['key']}"
            K = np.asarray(list(msg.k), dtype=np.float64).reshape(3, 3)
            D = np.asarray(
                [float(msg.d[i]) if i < len(msg.d) else 0.0 for i in range(8)],
                dtype=np.float64,
            )
            rfc_t = np.asarray(cam.rig_from_camera.translation)
            rfc_q = np.asarray(cam.rig_from_camera.rotation)
            print()
            print(f"--- [ros_oak_rgbd] {label} COLOR (CAM_A) ---")
            print(f"size = {cam.size}")
            print(f"focal = {cam.focal}")
            print(f"principal = {cam.principal}")
            print(f"K =\n{K}")
            print(f"D (8) = {D}")
            print(f"distortion model = {cam.distortion.model}")
            print(f"rig_from_camera.translation [m] = {rfc_t}")
            print(f"rig_from_camera.rotation (qx,qy,qz,qw) = {rfc_q}")
            print()

        rig = vslam.Rig()
        rig.cameras = cameras
        return rig

    def start_streaming(
        self, tracker: vslam.Tracker, output_queue: queue.Queue, **kwargs
    ) -> None:
        import message_filters
        from sensor_msgs.msg import Image
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._running = True
        n = len(self._entries)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Track-latency budget, logged every 1s. We have a ~33ms frame budget
        # (30 fps) to run tracker.track() and publish odometry on the single
        # spin thread. The only thing reported is: of the frames fed to track()
        # in the last second, how many stayed within budget vs blew past it,
        # and how much wall time the over-budget ones cost. That over-budget
        # wall time is exactly what's stolen from capturing the next frames,
        # since track() blocks this thread.
        #   track/s   = frames fed to tracker.track()
        #   <=33ms    = tracks within the frame budget
        #   >33ms     = tracks over budget
        #   sum       = accumulated wall time of the over-budget tracks
        BUDGET_MS = 1000.0 / 30.0  # ~33.3ms, one 30fps frame
        track_n  = [0]
        on_n     = [0]
        over_n   = [0]
        over_ms  = [0.0]
        last_log_time = [time.monotonic()]
        last_ts = [0]

        def _log_diag(now):
            if self._debug and now - last_log_time[0] >= 1.0:
                print(
                    f"[ros_oak_rgbd] track {track_n[0]}/s  "
                    f"<=33ms {on_n[0]}  >33ms {over_n[0]} (sum {over_ms[0]:.0f}ms)",
                    flush=True,
                )
                track_n[0] = on_n[0] = over_n[0] = 0
                over_ms[0] = 0.0
                last_log_time[0] = now

        def on_rgbd(*msgs):
            color_msgs = msgs[0::2]
            depth_msgs = msgs[1::2]
            ts = (
                color_msgs[0].header.stamp.sec * 1_000_000_000
                + color_msgs[0].header.stamp.nanosec
            )
            if ts <= last_ts[0]:
                return
            last_ts[0] = ts

            colors = [_decode_oak_raw_color(m) for m in color_msgs]
            if any(c is None for c in colors):
                bad = [self._entries[i]["key"]
                       for i, c in enumerate(colors) if c is None]
                print(f"[ros_oak_rgbd] color decode failed for {bad}", flush=True)
                return
            depths = [_decode_oak_raw_depth(m) for m in depth_msgs]
            if any(d is None for d in depths):
                bad = [self._entries[i]["key"]
                       for i, d in enumerate(depths) if d is None]
                print(f"[ros_oak_rgbd] depth decode failed for {bad}", flush=True)
                return

            t0 = time.monotonic()
            vo_pose_estimate, slam_pose = tracker.track(
                ts, images=colors, depths=depths
            )
            track_ms = (time.monotonic() - t0) * 1000

            track_n[0] += 1
            if track_ms > BUDGET_MS:
                over_n[0] += 1
                over_ms[0] += track_ms
            else:
                on_n[0] += 1
            _log_diag(time.monotonic())

            if vo_pose_estimate.world_from_rig is None or slam_pose is None:
                return

            landmarks = [
                Landmark(lm.id, lm.coords) for lm in tracker.get_last_landmarks()
            ]
            output_queue.put(
                TrackingResult(
                    ts,
                    Pose(vo_pose_estimate.world_from_rig.pose),
                    Pose(slam_pose),
                    tuple(colors),
                    landmarks,
                )
            )

        self._node = Node("ros_oak_rgbd_frames")

        subscribers = []
        topics_log: List[str] = []
        for entry in self._entries:
            for topic in (entry["color_topic"], entry["depth_topic"]):
                subscribers.append(
                    message_filters.Subscriber(
                        self._node, Image, topic, qos_profile=qos
                    )
                )
                topics_log.append(topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=QUEUE_SIZE, slop=SLOP_SEC
        )
        self._sync.registerCallback(on_rgbd)

        print(
            f"[ros_oak_rgbd] Subscribed (ApproximateTimeSynchronizer "
            f"slop={SLOP_SEC*1000:.0f}ms, {n} RGB-D pair(s)): "
            + ", ".join(topics_log),
            flush=True,
        )

        self._spin_thread = threading.Thread(
            target=_spin_ros_node, args=(self._node, self), daemon=True
        )
        self._spin_thread.start()

    def stop_streaming(self) -> None:
        self._running = False
        if hasattr(self, "_spin_thread"):
            self._spin_thread.join(timeout=2.0)
        if hasattr(self, "_node"):
            self._node.destroy_node()
