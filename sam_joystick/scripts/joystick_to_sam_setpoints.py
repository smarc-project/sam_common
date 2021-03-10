#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import rospy
import math

from sam_msgs.msg import ThrusterAngles
from smarc_msgs.msg import ThrusterRPM
from geometry_msgs.msg import Twist

class JoystickTranslator(object):
    def __init__(self):
        self.rpm_sub = rospy.Subscriber("ctrl/rpm_joystick", Twist, self.rpm_cb, queue_size=1)
        self.vector_sub = rospy.Subscriber("ctrl/vector_deg_joystick", Twist, self.vector_cb, queue_size=1)

        self.thruster1_pub = rospy.Publisher("core/thruster1_cmd", ThrusterRPM, queue_size=1)
        self.thruster2_pub = rospy.Publisher("core/thruster2_cmd", ThrusterRPM, queue_size=1)
        self.vector_pub = rospy.Publisher("core/thrust_vector_cmd", ThrusterAngles, queue_size=1)

        self.rpm_msg = ThrusterRPM()
        self.vec_msg = ThrusterAngles()

        self.published_zero_rpm_once = False
        self.published_zero_vec_once = False

    def rpm_cb(self, twist):
        # linear.x is forward, straight to RPMs
        self.rpm_msg.rpm = int(twist.linear.x)

    def vector_cb(self, twist):
        # angular.z is horizontal angle
        # angular.y is vertical angle
        hori = math.radians(twist.angular.z)
        vert = math.radians(twist.angular.y)

        self.vec_msg.thruster_horizontal_radians = hori
        self.vec_msg.thruster_vertical_radians = vert


    def publish(self):
        if self.rpm_msg.rpm != 0 or not self.published_zero_rpm_once:
            self.thruster1_pub.publish(self.rpm_msg)
            self.thruster2_pub.publish(self.rpm_msg)
            rospy.loginfo("Rpm:{}".format(self.rpm_msg.rpm))

            self.published_zero_rpm_once = self.rpm_msg.rpm == 0

        if self.vec_msg.thruster_horizontal_radians != 0 or \
           self.vec_msg.thruster_vertical_radians != 0 or \
           not self.published_zero_vec_once:
            self.vector_pub.publish(self.vec_msg)
            rospy.loginfo("Vec:{},{}".format(self.vec_msg.thruster_horizontal_radians, self.vec_msg.thruster_vertical_radians))

            self.published_zero_vec_once = self.vec_msg.thruster_horizontal_radians == 0 and\
                                           self.vec_msg.thruster_vertical_radians == 0

        # reset the stuff
        self.rpm_msg.rpm = 0
        self.vec_msg.thruster_horizontal_radians = 0
        self.vec_msg.thruster_vertical_radians = 0




if __name__ == "__main__":
    rospy.init_node("joystick_to_sam_setpoints")
    jt = JoystickTranslator()
    rospy.loginfo("Joystick translator for SAM is running...")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        jt.publish()
        rate.sleep()





