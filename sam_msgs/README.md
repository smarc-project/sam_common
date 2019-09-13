sam_drivers 
===========

This package contains messages that are used for the ROS/hardware interface in the SAM auv.

## Messages

* `ThrusterRPMs` - the target RPMs of the two thrusters as given in rot/min
* `ThrusterAngles` - the target vertical and horizontal angles of the thrusters as given in radians
* `BallastAngles` - the offset angles of the ballast weights as given in radians
* `PercentStamped` - a generic stamped float that is used for communicating buoyancy percent and offset percent of longitudinal weight
