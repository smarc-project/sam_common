# smarc_joy_ds5

[![CI](https://github.com/matthew-william-lock/sam_ds5_ros/actions/workflows/main.yaml/badge.svg)](https://github.com/matthew-william-lock/sam_ds5_ros/actions/workflows/main.yaml) [![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)

Control SAM AUV using a DualSense controller, and get sensory feedback from the controller's LED and vibration motors.

:warning: Does not support wireless bluetooth operation, use xbox controller for this instead.

<p align="center">
  <img src="https://user-images.githubusercontent.com/53016036/235476324-ffab01e0-7e11-438f-a3e0-7eedcea22abf.png" width="100%">
</p>

## Installation

> Steps for installation are taken from the [pydualsense](https://github.com/flok/pydualsense) dependency.

You first need to add a udev rule to let the user access the PS5 controller without requiring root privileges.

```bash
sudo cp 70-ps5-controller.rules /etc/udev/rules.d
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then install the ```libhidapi-dev```.

```
sudo apt install libhidapi-dev
```

After that install the package from [pypi](https://pypi.org/project/pydualsense/).

```bash
pip install pydualsense
```

## Launch

After the package has been built and the controller is connected, launch the node using the following command:

```bash
roslaunch sam_ds5_ros sam_ds5.launch
```

After launching, the following topics will be available:
```bash
- /sam/core/teleop/enable [std_msgs/Bool] (Publish status of teleop enable)
```

## Controls 

| Button | Action | Vibration Feedback | LED Feedback |
| --- | --- | --- | --- |
| ```Left stick``` | Send RPM commands to motors | - | - |
| ```Right stick``` | Send thrust vector commands | - | - |
| ```x``` | Toggle teleop enable | ```One short pulse``` - Enabled <br>```Two short pulse``` - Disabled | ```Green``` - Enabled<br>```Red``` - Disabled | 

## Status Indicators

| State | Vibration mode | LED colour |
| --- | --- | --- | 
| ```Leak detected``` | Continuous vibration | Flashing red |