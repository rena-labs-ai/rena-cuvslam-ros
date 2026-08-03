"""Launch the C++ cuVSLAM odometry node (rena_cuvslam_ros).

Supports both tracker modes (both read the base OAK cameras from
/etc/rena/config.yaml, single or multi-camera):
  tracker:=rgbd    (default) — RGBD odometry (color + aligned depth)
  tracker:=stereo            — Stereo odometry (raw left + right mono)

Assumes the OAK camera nodes are already running. Topics and rig extrinsics
are derived inside the node from /etc/rena/config.yaml.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    odom_topic = context.launch_configurations.get("odom_topic", "/cuvslam/odometry")
    planarize = context.launch_configurations.get(
        "planarize", "true").strip().lower() in ("1", "true", "yes", "on")
    publish_map_tf = context.launch_configurations.get(
        "publish_map_tf", "true").strip().lower() in ("1", "true", "yes", "on")
    log_level = context.launch_configurations.get("log_level", "info").strip().lower()
    debug = log_level == "debug"
    depth_scale = float(context.launch_configurations.get("depth_scale", "0.001"))

    return [
        Node(
            package="rena_cuvslam_ros",
            executable="vslam_node",
            name="vslam",
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                {
                    "odom_child_frame": LaunchConfiguration("odom_child_frame"),
                    "planarize": planarize,
                    "publish_map_tf": publish_map_tf,
                    "map_frame": LaunchConfiguration("map_frame"),
                    "debug": debug,
                    "tracker": LaunchConfiguration("tracker"),
                    "depth_scale": depth_scale,
                }
            ],
            remappings=[("/cuvslam/odometry", odom_topic)],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/cuvslam/odometry",
                description="Topic to publish odometry on (remapped from /cuvslam/odometry).",
            ),
            DeclareLaunchArgument(
                "odom_child_frame",
                default_value="base_nav_link",
                description="Odometry child_frame_id and odom -> child TF child frame; "
                "nav2's robot_base_frame.",
            ),
            DeclareLaunchArgument(
                "planarize",
                default_value="true",
                description="Planar mode for the published TF (zero VSLAM roll/pitch on "
                "odom -> base_nav_link). The /cuvslam/odometry topic stays raw 6-DOF.",
            ),
            DeclareLaunchArgument(
                "map_frame",
                default_value="map",
                description="Static map -> odom parent frame.",
            ),
            DeclareLaunchArgument(
                "publish_map_tf",
                default_value="true",
                description="Publish the map -> odom backend correction TF. Set "
                "false when an external backend (e.g. rtabmap) owns map -> odom.",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="Global ROS log level; 'debug' enables the per-second "
                "tracker diagnostics ladder.",
            ),
            DeclareLaunchArgument(
                "tracker",
                default_value="rgbd",
                description="Tracker mode: 'rgbd' (color + aligned depth) or "
                "'stereo' (raw left/right mono); both use the base OAK cameras.",
            ),
            DeclareLaunchArgument(
                "depth_scale",
                default_value="0.001",
                description="Depth scale factor for RGBD mode (meters per raw depth unit; "
                "0.001 = mm -> m). Ignored in stereo mode.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
