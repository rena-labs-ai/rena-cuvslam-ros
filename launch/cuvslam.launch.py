"""Launch the C++ cuVSLAM RGB-D odometry node (rena_cuvslam_ros_cpp).

Drop-in for the RGBD path of rena_cuvslam_ros's cuvslam.launch.py. Assumes the
base OAK camera(s) are already running. Topics are derived inside the node from
/etc/rena/config.yaml (base OAK serials/keys -> rgb + stereo image_raw).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    odom_topic = context.launch_configurations.get("odom_topic", "/cuvslam/odometry")
    planarize = context.launch_configurations.get(
        "planarize", "true").strip().lower() in ("1", "true", "yes", "on")
    log_level = context.launch_configurations.get("log_level", "info").strip().lower()
    debug = log_level == "debug"

    return [
        Node(
            package="rena_cuvslam_ros_cpp",
            executable="vslam_node",
            name="vslam",
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                {
                    "odom_child_frame": LaunchConfiguration("odom_child_frame"),
                    "planarize": planarize,
                    "map_frame": LaunchConfiguration("map_frame"),
                    "debug": debug,
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
                "log_level",
                default_value="info",
                description="Global ROS log level; 'debug' enables the per-second "
                "tracker diagnostics ladder.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
