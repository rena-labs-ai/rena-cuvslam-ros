"""ROS2 node that runs cuVSLAM on an OAK camera and publishes odometry to
/cuvslam/odometry.

Supports the OAK trackers: RosOakStereoTracker and RosOakRGBDTracker. Image
topics are derived inside the tracker from /etc/rena/config.yaml
(serial + image_mode -> image_raw | image_rect).

Publishes raw 6-DOF Odometry on /cuvslam/odometry AND broadcasts the
odom -> base_nav_link TF (planarized) directly, plus the static map -> odom.
Stamps match the tracker pipeline timestamp (synced left header time from
ApproximateTimeSynchronizer), which is the stamp nvblox looks up at depth time,
so owning the TF here avoids a separate node re-subscribing to the odom topic
just to re-emit it. This module holds all ROS wiring; tracking logic lives in
tracker.py / pipeline.py.
"""

import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    Quaternion,
    TransformStamped,
    TwistWithCovariance,
)
from nav_msgs.msg import Odometry
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .pipeline import Pipeline
from .plot import compute_ate, plot_combined
from .tracker import RosOakRGBDTracker, RosOakStereoTracker

ODOM_TOPIC = "/cuvslam/odometry"
ODOM_FRAME = "odom"


def _stamp_from_ns(timestamp_ns: int) -> Time:
    t = Time()
    t.sec = int(timestamp_ns // 1_000_000_000)
    t.nanosec = int(timestamp_ns % 1_000_000_000)
    return t


def _make_tracker(tracker_type: str):
    match tracker_type:
        case "ros_oak_stereo":
            return RosOakStereoTracker()
        case "ros_oak_rgbd":
            return RosOakRGBDTracker()
        case _:
            raise ValueError(
                f"Unknown tracker type: {tracker_type!r}; "
                "supported: 'ros_oak_stereo', 'ros_oak_rgbd'"
            )


class VslamNode(Node):
    def __init__(
        self,
        tracker,
        child_frame: str,
        planarize: bool = True,
        map_frame: str = "map",
    ) -> None:
        super().__init__("vslam")
        self._child_frame = child_frame
        self._planarize = planarize
        self._odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)

        # cuVSLAM owns the odom -> child_frame TF directly: the pose and the
        # stereo-left stamp are already in this loop, so broadcasting here avoids
        # a separate node round-tripping /cuvslam/odometry just to re-emit it as
        # TF. The odometry TOPIC stays raw 6-DOF; only the TF is planarized (zero
        # VSLAM roll/pitch -- mount tilt lives in the static base_nav_link ->
        # camera rig TF, so stock nvblox integrates at the calibrated pose).
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_broadcaster = StaticTransformBroadcaster(self)
        st = TransformStamped()
        st.header.stamp = self.get_clock().now().to_msg()
        st.header.frame_id = map_frame
        st.child_frame_id = ODOM_FRAME
        st.transform.rotation.w = 1.0
        self._static_broadcaster.sendTransform(st)

        self._pipeline = Pipeline(tracker)
        self._pipeline.start()

        mode = "PLANAR (yaw only)" if planarize else "full 6-DOF"
        self.get_logger().info(
            f"Publishing odometry on {ODOM_TOPIC} (raw 6-DOF) + TF "
            f"{ODOM_FRAME}->{child_frame} [{mode}]; static {map_frame}->{ODOM_FRAME}"
        )
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

    def _tracking_loop(self) -> None:
        while rclpy.ok():
            result = self._pipeline.get(timeout=1.0)
            if result is None or result.slam_pose is None:
                continue
            stamp = _stamp_from_ns(result.timestamp)
            pose_robot = result.slam_pose.to_robot_frame()
            t = pose_robot.translation
            r = pose_robot.rotation

            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = ODOM_FRAME
            msg.child_frame_id = self._child_frame
            msg.pose = PoseWithCovariance()
            msg.pose.pose = Pose()
            msg.pose.pose.position = Point(x=float(t[0]), y=float(t[1]), z=float(t[2]))
            msg.pose.pose.orientation = Quaternion(
                x=float(r[0]), y=float(r[1]), z=float(r[2]), w=float(r[3])
            )
            msg.twist = TwistWithCovariance()
            self._odom_pub.publish(msg)

            # Same pose as TF at the same stamp. Topic stays raw 6-DOF above;
            # planarize only the TF (r is [x, y, z, w]).
            ts = TransformStamped()
            ts.header.stamp = stamp
            ts.header.frame_id = ODOM_FRAME
            ts.child_frame_id = self._child_frame
            ts.transform.translation.x = float(t[0])
            ts.transform.translation.y = float(t[1])
            ts.transform.translation.z = float(t[2])
            if self._planarize:
                yaw = math.atan2(
                    2.0 * (r[3] * r[2] + r[0] * r[1]),
                    1.0 - 2.0 * (r[1] * r[1] + r[2] * r[2]),
                )
                half = 0.5 * yaw
                ts.transform.rotation.z = math.sin(half)
                ts.transform.rotation.w = math.cos(half)
            else:
                ts.transform.rotation.x = float(r[0])
                ts.transform.rotation.y = float(r[1])
                ts.transform.rotation.z = float(r[2])
                ts.transform.rotation.w = float(r[3])
            self._tf_broadcaster.sendTransform(ts)

    def destroy_node(self) -> None:
        self._pipeline.stop()
        super().destroy_node()


def main() -> None:
    rclpy.init()

    param_node = Node("vslam")
    tracker_param = param_node.declare_parameter("tracker", "ros_oak_rgbd")
    odom_child_frame_param = param_node.declare_parameter(
        "odom_child_frame", "base_nav_link"
    )
    planarize_param = param_node.declare_parameter("planarize", True)
    map_frame_param = param_node.declare_parameter("map_frame", "map")
    tracker_type = str(tracker_param.value)
    odom_child_frame = str(odom_child_frame_param.value)
    planarize = bool(planarize_param.value)
    map_frame = str(map_frame_param.value)
    param_node.destroy_node()

    # OAK topics are derived from /etc/rena/config.yaml inside the tracker
    # (serial + image_mode -> image_raw | image_rect).
    tracker = _make_tracker(tracker_type)

    node = VslamNode(
        tracker,
        child_frame=odom_child_frame,
        planarize=planarize,
        map_frame=map_frame,
    )
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Optional odometry plotter (enable_plot): compares the cuVSLAM estimate against
# a reference odom topic and periodically writes a trajectory PNG. Separate node
# so it only runs when requested; plotting/metrics live in plot.py.
# --------------------------------------------------------------------------


def _quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomLogger(Node):
    """Sample a reference and an estimated odom topic at 1 Hz, render ref-vs-estimate."""

    def __init__(self) -> None:
        super().__init__("odom_logger")
        self.declare_parameter("ref_odom_topic", "/Odometry")
        self.declare_parameter("estimated_odom_topic", ODOM_TOPIC)
        self.declare_parameter("estimated_label", "cuvslam")
        self.declare_parameter("out_path", "./outputs/vslam_plot.png")
        self.declare_parameter("title", "Odometry Comparison")
        self.declare_parameter("update_interval_sec", 2.0)

        gp = lambda n: self.get_parameter(n).get_parameter_value()
        ref_topic = gp("ref_odom_topic").string_value
        est_topic = gp("estimated_odom_topic").string_value
        self._label = gp("estimated_label").string_value or est_topic
        self._out_path = Path(gp("out_path").string_value)
        self._title = gp("title").string_value
        update_interval = gp("update_interval_sec").double_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._last_ref: Optional[Odometry] = None
        self._last_est: Optional[Odometry] = None
        self._ref_traj: list = []
        self._est_traj: list = []

        self.create_subscription(Odometry, ref_topic, self._on_ref, qos)
        self.create_subscription(Odometry, est_topic, self._on_est, qos)
        self.get_logger().info(
            f"odom plot: ref={ref_topic} estimated={est_topic} -> {self._out_path}"
        )
        self.create_timer(1.0, self._sample)
        self.create_timer(update_interval, self._render)

    def _on_ref(self, msg: Odometry) -> None:
        self._last_ref = msg

    def _on_est(self, msg: Odometry) -> None:
        self._last_est = msg

    def _sample(self) -> None:
        if self._last_est is None:
            return
        if self._last_ref is not None:
            r = self._last_ref.pose.pose
            self._ref_traj.append((r.position.x, r.position.y, _quat_to_yaw(r.orientation)))
        c = self._last_est.pose.pose
        self._est_traj.append((c.position.x, c.position.y, _quat_to_yaw(c.orientation)))

    def _render(self) -> None:
        if len(self._est_traj) < 2:
            return
        try:
            plot_combined(self._ref_traj, [(self._label, self._est_traj)], self._title, self._out_path)
        except Exception as e:
            self.get_logger().warn(f"Plot update failed: {e}")

    def report(self) -> None:
        if len(self._est_traj) < 2:
            print(f"[odom_logger] only {len(self._est_traj)} sample(s), skipping", file=sys.stderr)
            return
        try:
            plot_combined(self._ref_traj, [(self._label, self._est_traj)], self._title, self._out_path)
            print(f"[odom_logger] final plot saved to {self._out_path}", file=sys.stderr)
        except Exception as e:
            print(f"[odom_logger] failed to save final plot: {e}", file=sys.stderr)
        if self._ref_traj:
            ate_rmse, _ = compute_ate(self._ref_traj, self._est_traj)
            print(f"[odom_logger] ATE RMSE = {ate_rmse:.4f} m", file=sys.stderr)


def plot_main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    node.report()
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
