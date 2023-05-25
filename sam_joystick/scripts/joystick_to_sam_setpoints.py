#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import rospy
import math

from sam_msgs.msg import ThrusterAngles, ThrusterRPMs
from geometry_msgs.msg import Twist

class JoystickTranslator(object):
    def __init__(self):

        rpm_cmd_top = rospy.get_param("~rpm_cmd_top", "/rpm_cmd")
        thrust_cmd_top = rospy.get_param("~thrust_vector_cmd_top", "/thrust_vector_cmd")
        rpm_joystick_top = rospy.get_param("~rpm_joystick_top", "/rpm_joystick")
        vector_deg_joystick_top = rospy.get_param("~vector_deg_joystick_top", "/vector_deg_joystick")
        freq = rospy.get_param("~node_freq", 1)

        self.rpm_sub = rospy.Subscriber(rpm_joystick_top, Twist, self.rpm_cb, queue_size=1)
        self.vector_sub = rospy.Subscriber(vector_deg_joystick_top, Twist, self.vector_cb, queue_size=1)

        self.thruster_pub = rospy.Publisher(rpm_cmd_top, ThrusterRPMs, queue_size=1)
        self.vector_pub = rospy.Publisher(thrust_cmd_top, ThrusterAngles, queue_size=1)

        self.rpm_msg = ThrusterRPMs()
        self.vec_msg = ThrusterAngles()

        self.published_zero_rpm_once = False
        self.published_zero_vec_once = False

        rate = rospy.Rate(freq)
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()

    rospy.loginfo("Joystick listener stopped")

    def rpm_cb(self, twist):
        # linear.x is forward, straight to RPMs
        self.rpm_msg.thruster_1_rpm = int(twist.linear.x)
        self.rpm_msg.thruster_2_rpm = int(twist.linear.x)

    def vector_cb(self, twist):
        hori = math.radians(twist.angular.z)
        vert = math.radians(twist.angular.y)

        self.vec_msg.thruster_horizontal_radians = hori
        self.vec_msg.thruster_vertical_radians = vert


    def publish(self):
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

        # reset the stuff
        # Legacy code, unsure if needed for ROS-Mobile-Android
        # self.rpm_msg.thruster_1_rpm = 0
        # self.rpm_msg.thruster_2_rpm = 0
        # self.vec_msg.thruster_horizontal_radians = 0
        # self.vec_msg.thruster_vertical_radians = 0


if __name__ == "__main__":
    rospy.init_node("joystick_to_sam_setpoints")
    jt = JoystickTranslator()
    rospy.loginfo("Digital joystick translator for SAM is running...")





