# rena_cuvslam_ros

ROS 2 node that runs NVIDIA cuVSLAM on OAK RGB-D cameras and publishes visual
odometry on `/cuvslam/odometry`. Calls the cuVSLAM C++ API (`cuvslam2.h`)
directly — no Python or pybind overhead — so all camera streams are drained at
full 30 fps.

## Layout

```
src/
  rgbd_tracker.cpp   # subscription, ApproximateTime sync, zero-copy Track()
  vslam_node.cpp     # ROS node: odometry publisher + TF broadcaster
include/rena_cuvslam_ros/
  rgbd_tracker.hpp
  frame_conversions.hpp  # cuVSLAM optical <-> ROS frame math (pure, no deps)
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
ros2 launch rena_cuvslam_ros cuvslam.launch.py
```

### Launch arguments

| arg                | default             | description                                      |
|--------------------|---------------------|--------------------------------------------------|
| `odom_topic`       | `/cuvslam/odometry` | topic the odometry is published on               |
| `odom_child_frame` | `base_nav_link`     | `child_frame_id` and odom → child TF child frame |
| `planarize`        | `true`              | zero roll/pitch on the odom TF (yaw only)        |
| `map_frame`        | `map`               | parent frame for the static map → odom TF        |
| `log_level`        | `info`              | ROS log level                                    |
