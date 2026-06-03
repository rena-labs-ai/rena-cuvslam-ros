# rena_cuvslam_ros

ROS 2 node that runs NVIDIA cuVSLAM on an OAK camera and publishes visual
odometry on `/cuvslam/odometry`. Self-contained ament_python package — no
`cuvslam_examples` dependency.

## Layout

```
rena_cuvslam_ros/
  tracker.py    # OAK stereo / RGBD cuVSLAM trackers + shared tracking logic (pure)
  pipeline.py   # generic cuVSLAM tracking pipeline (pure)
  node.py       # all ROS wiring: VslamNode (odometry) + OdomLogger (optional plot)
  plot.py       # trajectory metrics + plotting, no ROS (unit-testable)
launch/
  cuvslam.launch.py
```

Pure logic lives in `tracker.py` / `pipeline.py` / `plot.py`; everything that
touches rclpy is in `node.py`.

## Dependencies

Install the cuVSLAM Python binding (match your arch / CUDA):

```bash
# aarch64
pip install https://github.com/nvidia-isaac/cuVSLAM/releases/download/v15.0.0/cuvslam-15.0.0+cu13-cp312-abi3-manylinux_2_39_aarch64.whl
# x86_64
pip install https://github.com/nvidia-isaac/cuVSLAM/releases/download/v15.0.0/cuvslam-15.0.0+cu13-cp312-abi3-manylinux_2_39_x86_64.whl
```

Other deps (`rclpy`, `sensor_msgs`, `message_filters`, `python3-opencv`,
`matplotlib`) are declared in `package.xml`.

## Build

```bash
colcon build --packages-select rena_cuvslam_ros
```

## Run

Only OAK cameras are supported. Camera topics are derived inside the tracker
from `/etc/rena/config.yaml` (the robot config written by rena-commission):
each OAK entry's `serial_no` + `image_mode` → `image_raw` | `image_rect`, under
the `/<robot_part>/<key>/...` namespace (e.g. `/base/front/...`). There is no
separate rig config to edit.

1. Bring up the OAK camera (via the robot bring-up, e.g. `rena start`, which
   launches the `depthai_ros_driver_v3` OAK driver).

2. Launch the node:

```bash
# RGBD tracker (default)
ros2 launch rena_cuvslam_ros cuvslam.launch.py

# or the stereo tracker
ros2 launch rena_cuvslam_ros cuvslam.launch.py tracker:=ros_oak_stereo
```

### Launch arguments

| arg                | default              | description                                            |
| ------------------ | -------------------- | ------------------------------------------------------ |
| `tracker`          | `ros_oak_rgbd`       | `ros_oak_rgbd` or `ros_oak_stereo`                     |
| `odom_topic`       | `/cuvslam/odometry`  | topic the odometry is published on                     |
| `odom_child_frame` | `base_camera_link`   | `child_frame_id` of the published odometry             |
| `enable_plot`      | `false`              | also start `odom_logger` (see below)                   |
| `ref_odom_topic`   | `/Odometry`          | reference odom for the plot (e.g. fast_lio)            |
| `plot_out_path`    | `./outputs/vslam_plot.png` | output PNG path                                  |
| `experiment`       | `Odometry Comparison`| plot title                                             |

### Plotting

With `enable_plot:=true` an `odom_logger` node samples `/cuvslam/odometry`
against a reference odom topic (default `/Odometry`, e.g. fast_lio) at 1 Hz and
periodically writes a trajectory + ATE PNG to `plot_out_path` (a final plot and
ATE RMSE are emitted on shutdown).
