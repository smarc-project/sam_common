# SAM Joystick Controls
There is a single node in here, sam_joystick, that listens to `/sam/ctrl/rpm_joystick` (geometry_msgs/Twist) and `/sam/ctrl/vector_deg_joystick` (also Twist). 
The rpm topic only uses the `linear.x` component and publishes that to both thrusters of SAM directly as RPM.
The vec topic uses the `angular.z` and `angular.y` fields, converts them to radians and publishes to thrust vectoring. 

This node was tested with joysticks from this app: https://github.com/ROS-Mobile/ROS-Mobile-Android.

## App setup
1- Follow the FAQ in ROS-Mobile-Android and setup your IP addresses and such both in the vehicle and your phone.
Copied here:
```
export ROS_IP=SYSTEM_IP
export ROS_MASTER_URI=http://$ROS_IP:11311
export ROS_HOSTNAME=$ROS_IP
```
Where SYSTEM_IP is the IP of the master. Put these into .bashrc on the master.

2- In the app, go to MASTER tab and put the IP address of the master and press connect. The button should turn red.

3- Go to DETAILS tab and press plus. Insert a Joystick.

4- Set these for the RPM joystick: 
```
x:0
y:0
Width:4
Height:4
topic_name:sam/ctrl/rpm_joystick
x-axis: angular Z with scale 0 to 0
y-axis: linear X with scale -1000 to 1000 (or whatever RPM range you want to use)
```

5- Press the 'refresh' arrow near top-right.

6- Insert a second Joystick for thrust-vectoring: 
```
x:4
y:0
width:4
height:4
topic-name:sam/ctrl/vector_deg_joystick
x-axis: Angular Z with scale 5 to -5
y-axis: Angular Y with scale 5 to -5
```
(the x and y axis scales can be swapped if you prefer inverted controls)

7- Press the refresh arrow of this joystick.

8- Build and source if needed, then: `roslaunch sam_joystick sam_joystick.launch`

9- Drive responsibly.


It should look like this: https://www.dropbox.com/s/zi9e80mnb8fi0sw/20210310_101710.mp4?dl=0

## Things missing:
- Configurable ranges in the node, in case ranges of the joystick itself (like the app allows) can not be configured.
- Different topics?
- Build stuff for release (CMakeLists dependencies on sam_msgs and smarc_msgs etc.)
