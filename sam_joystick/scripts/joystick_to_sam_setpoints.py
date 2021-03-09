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

    def rpm_cb(self, twist):
        rospy.loginfo("xy vector:{}".format(twist))
        # linear.x is forward, straight to RPMs
        rpm = int(twist.linear.x)
        rpm_msg = ThrusterRPM()
        rpm_msg.rpm = rpm

        self.thruster1_pub.publish(rpm_msg)
        self.thruster2_pub.publish(rpm_msg)



    def vector_cb(self, twist):
        rospy.loginfo("z vector:{}".format(twist))

        # angular.z is horizontal angle
        # angular.y is vertical angle
        hori = math.radians(twist.angular.z)
        vert = math.radians(twist.angular.y)

        vec_msg = ThrusterAngles()
        vec_msg.thruster_horizontal_radians = hori
        vec_msg.thruster_vertical_radians = vert

        self.vector_pub.publish(vec_msg)




if __name__ == "__main__":
    rospy.init_node("joystick_to_sam_setpoints")
    jt = JoystickTranslator()
    rospy.loginfo("Joystick translator for SAM is running...")
    rospy.spin()




