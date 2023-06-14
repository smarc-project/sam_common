#!/usr/bin/env python3

"""
MIT License

Copyright (c) 2023 Matthew Lock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Create a ROS node determines the setpoints for the SAM thrusters based on the joystick input


import rospy

from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import Twist
from sam_msgs.msg import ThrusterAngles, ThrusterRPMs

from sam_joy_msgs.msg import JoyButtons

import math

class teleop():
    
    # ================================================================================
    # Callback Functions
    # ================================================================================

    def joy_btns_callback(self, msg:JoyButtons):
        """
        Callback function for the joystick subscriber
        """

        RPM_MAX = 1500
        RAD_MAX = 0.1
        RPM_LINEAR_STEP_SIZE = 1/15

        RAD_STEPS = 5
        RAD_STEP_SIZE = RAD_MAX / RAD_STEPS
        LINEAR_STEP_SIZE = 1/RAD_STEPS

        if self.teleop_enabled:

            rpm_cmd = int(msg.left_y* RPM_MAX)
            x_cmd = msg.right_x * RAD_MAX
            y_cmd = msg.right_y * RAD_MAX

            # Round rpm_cmd to nearest 100
            rpm_cmd = int(round(rpm_cmd, -2))

            # Round x_cmd and y_cmd to nearest value on range RAD_MIN to RAD_MAX with RAD_STEPS steps
            x_steps = round(x_cmd / RAD_STEP_SIZE)
            x_cmd = x_steps * RAD_STEP_SIZE
            y_steps = round(y_cmd / RAD_STEP_SIZE)
            y_cmd = y_steps * RAD_STEP_SIZE

            
            # If assisted depth-keeping, ctrl_msg.angular.y is overriden by PID controller
            y_cmd = max(min(y_cmd, 0.1), -0.1)  # Elevator boundaries
            elev_cmd = self.elev_effort if self.assisted_driving_enabled else y_cmd

            self.rpm_msg.thruster_1_rpm = int(rpm_cmd)
            self.rpm_msg.thruster_2_rpm = int(rpm_cmd)
            self.vec_msg.thruster_horizontal_radians = - x_cmd
            self.vec_msg.thruster_vertical_radians = elev_cmd




    def send_cmds(self):
        if self.rpm_msg.thruster_1_rpm != 0 or not self.published_zero_rpm_once:
            
            self.thruster_pub.publish(self.rpm_msg)
            zero = self.rpm_msg.thruster_1_rpm == 0
            
            if zero:
                rospy.loginfo(">>>>> Published 0 RPM")
            else:
                rospy.loginfo_throttle(1, "Rpm:{}".format(self.rpm_msg.thruster_1_rpm))

            self.published_zero_rpm_once = zero

        if self.vec_msg.thruster_horizontal_radians != 0 or \
           self.vec_msg.thruster_vertical_radians != 0 or \
           not self.published_zero_vec_once:

            self.vector_pub.publish(self.vec_msg)

            zero = self.vec_msg.thruster_horizontal_radians == 0 and\
                   self.vec_msg.thruster_vertical_radians == 0
            
            if zero:
                rospy.loginfo(">>>>> Published 0 thrust vector")
            else:
                rospy.loginfo_throttle(1, "Vec:{},{}".format(self.vec_msg.thruster_horizontal_radians, self.vec_msg.thruster_vertical_radians))

            self.published_zero_vec_once = zero


            
    def teleop_enabled_callback(self, msg: Bool):
        """
        Callback function for the teleop enabled subscriber
        """
        self.teleop_enabled = msg.data

    def assisted_driving_callback(self, msg: Bool):
        """
        Callback function for the assisted driving subscriber
        """
        self.assisted_driving_enabled = msg.data
        self.start_ad = msg.data

    def depth_cb(self, msg):

        if self.assisted_driving_enabled:

            # Lock current depth when assisted driving is turned on to
            # be used as PID setpoint
            if self.start_ad:
                self.depth_sp = msg.data
                self.start_ad = False
        
            self.elev_sp_pub.publish(Float64(self.depth_sp))

    def elev_pid_cb(self, msg):

        self.elev_effort = msg.data

    # ================================================================================
    # Node Init
    # ================================================================================

    def __init__(self):

        # pub = rospy.Publisher('chatter', String, queue_size=10)
        rospy.init_node('ds5_teleop', anonymous=True)

        rpm_joystick_top = rospy.get_param("~rpm_joystick_top", "/rpm_joystick")
        vector_deg_joystick_top = rospy.get_param("~vector_deg_joystick_top", "/vector_deg_joystick")
        teleop_enable_top = rospy.get_param("~teleop_enable", "/enable")
        assist_enable_top = rospy.get_param("~assist_enable", "/assist")
        joy_buttons_top = rospy.get_param("~joy_buttons", "/joy_buttons")
        
        depth_top = rospy.get_param("~depth_top", "/depth")
        elevator_pid_top = rospy.get_param("~elevator_pid_ctrl", "/elevator")
        elev_sp_top = rospy.get_param("~elev_sp_top")

        rpm_cmd_top = rospy.get_param("~rpm_cmd_top", "/rpm_cmd")
        thrust_cmd_top = rospy.get_param("~thrust_vector_cmd_top", "/thrust_vector_cmd")

        self.depth_sp = 0.
        self.elev_effort = 0.
        self.start_ad = True

        self.rpm_msg = ThrusterRPMs()
        self.vec_msg = ThrusterAngles()

        self.published_zero_rpm_once = False
        self.published_zero_vec_once = False

        # States
        self.teleop_enabled = False
        self.assisted_driving_enabled = False

        # Publishers
        self.rpm_joystick_pub = rospy.Publisher(rpm_joystick_top, Twist, queue_size=1)
        self.vector_deg_joystick_pub = rospy.Publisher(vector_deg_joystick_top, Twist, queue_size=1)
        self.elev_sp_pub = rospy.Publisher(elev_sp_top, Float64, queue_size=1)
        self.thruster_pub = rospy.Publisher(rpm_cmd_top, ThrusterRPMs, queue_size=1)
        self.vector_pub = rospy.Publisher(thrust_cmd_top, ThrusterAngles, queue_size=1)

        # Subscribers
        self.joy_btn_sub = rospy.Subscriber(joy_buttons_top, JoyButtons, self.joy_btns_callback)
        self.teleop_enabled_sub = rospy.Subscriber(teleop_enable_top, Bool, self.teleop_enabled_callback)
        self.assit_driving_sub = rospy.Subscriber(assist_enable_top, Bool, self.assisted_driving_callback)
        self.depth_sub = rospy.Subscriber(depth_top, Float64, self.depth_cb)
        self.elevator_pid_sub = rospy.Subscriber(elevator_pid_top, Float64, self.elev_pid_cb)

        rate = rospy.Rate(12)
        while not rospy.is_shutdown():
            self.send_cmds()
            rate.sleep()

        # rospy.spin()
  

if __name__ == '__main__':
    try:
        teleop()
    except rospy.ROSInterruptException:
        pass