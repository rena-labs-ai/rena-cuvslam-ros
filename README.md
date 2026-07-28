# rena_cuvslam_ros

ROS 2 node that runs NVIDIA cuVSLAM on OAK cameras and publishes visual
odometry on `/cuvslam/odometry`. Calls the cuVSLAM C++ API (`cuvslam2.h`)
directly — no Python or pybind overhead — so all camera streams are drained at
full 30 fps.

TF output follows the REP-105 split: the frontend (VO) pose drives
`odom → base_nav_link` (continuous, drifts), and the backend (loop-closure)
correction drives `map → odom` (jumps on closure), so `map → base_nav_link`
composes to the backend pose. The odometry topic carries the raw 6-DOF VO pose.

Supports two tracker modes selectable at launch time:

- **`rgbd`** (default) — RGB-D odometry using base OAK RGB + depth
- **`stereo`** — stereo odometry using base OAK cameras (raw left + right)

## Layout

```
src/
  rgbd_tracker.cpp      # RGBD path: ApproximateTime sync, zero-copy Track()
  stereo_tracker.cpp    # Stereo path: same sync structure, MONO8 images
  tracker_common.cpp    # Shared /etc/rena/config.yaml base-camera parsing
  vslam_node.cpp        # ROS node: odometry publisher + TF broadcaster
include/rena_cuvslam_ros/
  rgbd_tracker.hpp
  stereo_tracker.hpp
  tracker_common.hpp    # OakCameraConfig + LatestSlot<T> + CameraStatsLogger
  frame_conversions.hpp # cuVSLAM optical <-> ROS frame math (pure, no deps)
launch/
  cuvslam.launch.py
```

## Dependencies

Requires the cuVSLAM C++ library built from source (the `rena-control` Ansible
role handles this via `build_and_install.sh`). ROS deps are declared in
`package.xml`.

## Build

```bash
colcon build --merge-install --packages-select rena_cuvslam_ros
```

## Run

Camera topics and rig extrinsics are read from `/etc/rena/config.yaml` (written
by rena-commission). Start the OAK cameras first, then:

```bash
# RGBD (default)
ros2 launch rena_cuvslam_ros cuvslam.launch.py

# Stereo
ros2 launch rena_cuvslam_ros cuvslam.launch.py tracker:=stereo
```

### Launch arguments

| arg                | default             | description                                           |
|--------------------|---------------------|-------------------------------------------------------|
| `tracker`          | `rgbd`              | `rgbd` or `stereo`                                    |
| `odom_topic`       | `/cuvslam/odometry` | topic the odometry is published on                    |
| `odom_child_frame` | `base_nav_link`     | `child_frame_id` and odom → child TF child frame      |
| `planarize`        | `true`              | both TFs true planar: x, y, yaw only (z/roll/pitch zeroed) |
| `map_frame`        | `map`               | parent frame for the map → odom correction TF         |
| `depth_scale`      | `0.001`             | depth unit → metres (RGBD only; mm→m = 0.001)        |
| `log_level`        | `info`              | ROS log level; `debug` enables per-second diagnostics |

## Odometry plot utility

`.maps/vslam_plot.py` compares cuVSLAM output against a reference odometry
topic and computes ATE. Runs standalone (source ROS, no package install needed):

```bash
python3 .maps/vslam_plot.py --ref /base/drive_controller/odom --est /cuvslam/odometry \
    --out ./vslam_plot.png --update-interval 5.0
```
