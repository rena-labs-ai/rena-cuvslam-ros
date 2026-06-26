"""Launch the cuVSLAM odometry node. Assumes the OAK camera is already running.

With enable_plot:=true, also starts an odom_logger node that compares the
cuVSLAM estimate against a reference odom topic and writes a trajectory PNG.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    odom_topic = context.launch_configurations.get("odom_topic", "/cuvslam/odometry")
    enable_plot = context.launch_configurations.get("enable_plot", "false").lower() == "true"
    planarize = context.launch_configurations.get(
        "planarize", "true").strip().lower() in ("1", "true", "yes", "on")
    debug = context.launch_configurations.get(
        "log_level", "info").strip().lower() == "debug"

    actions = [
        Node(
            package="rena_cuvslam_ros",
            executable="vslam_node",
            name="vslam",
            output="screen",
            parameters=[
                {
                    "tracker": LaunchConfiguration("tracker"),
                    "odom_child_frame": LaunchConfiguration("odom_child_frame"),
                    "planarize": planarize,
                    "debug": debug,
                }
            ],
            remappings=[("/cuvslam/odometry", odom_topic)],
        ),
    ]

    if enable_plot:
        actions.append(
            Node(
                package="rena_cuvslam_ros",
                executable="odom_logger",
                name="odom_logger",
                output="screen",
                parameters=[
                    {
                        "ref_odom_topic": LaunchConfiguration("ref_odom_topic"),
                        "estimated_odom_topic": odom_topic,
                        "estimated_label": LaunchConfiguration("tracker"),
                        "out_path": LaunchConfiguration("plot_out_path"),
                        "title": LaunchConfiguration("experiment"),
                    }
                ],
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "tracker",
                default_value="ros_oak_rgbd",
                description="OAK tracker id.",
                choices=["ros_oak_stereo", "ros_oak_rgbd"],
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/cuvslam/odometry",
                description="Topic to publish odometry on (remapped from /cuvslam/odometry).",
            ),
            DeclareLaunchArgument(
                "odom_child_frame",
                default_value="base_nav_link",
                description="Odometry child_frame_id (robot nav center) and the "
                "odom -> child TF child frame; nav2's robot_base_frame.",
            ),
            DeclareLaunchArgument(
                "planarize",
                default_value="true",
                description="Planar mode for the published TF: zero VSLAM "
                "roll/pitch on odom -> base_nav_link (mount tilt comes from the "
                "rig TF). The /cuvslam/odometry topic stays raw 6-DOF either way. "
                "Set false for raw 6-DOF TF passthrough.",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="Global ROS log level; 'debug' enables the "
                "per-second tracker diagnostics log.",
            ),
            DeclareLaunchArgument(
                "enable_plot",
                default_value="false",
                description="Start odom_logger to plot cuVSLAM vs a reference odom topic.",
            ),
            DeclareLaunchArgument(
                "ref_odom_topic",
                default_value="/Odometry",
                description="Reference odometry topic for the plot.",
            ),
            DeclareLaunchArgument(
                "plot_out_path",
                default_value="./outputs/vslam_plot.png",
                description="Output PNG path for the tracker-vs-ref plot.",
            ),
            DeclareLaunchArgument(
                "experiment",
                default_value="Odometry Comparison",
                description="Plot title / experiment name.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
