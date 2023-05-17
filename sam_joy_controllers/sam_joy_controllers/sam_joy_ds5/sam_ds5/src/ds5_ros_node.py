#!/usr/bin/env python3

# THIS CODE IS ADAPTED FROM https://github.com/autonohm/ds5_ros

import rospy
import math
from sensor_msgs.msg import JoyFeedbackArray, Joy
from pydualsense import *

from ds5_msgs.msg import SetColour
from ds5_msgs.msg import SetMotor
from ds5_msgs.msg import JoyButtons

from sam_msgs.msg import Leak

class Ds5Ros():
    def __init__(self):
        self.logging_prefix = "DS5_ROS: "
        # 0 is startup, 1 is running
        self.node_state = 0
        # create dualsense
        self.dualsense = pydualsense()
        self.noderate = rospy.get_param("noderate", 50.0)
        # receive feedback from robot_joy_control node
        self.joy_sub_topic = rospy.get_param("joy_sub", "joy/set_feedback")
        
        # send signal to robot_joy_control node
        self.joy_pub_topic = rospy.get_param("joy_pub", "joy")
        self.btn_pres_topic = rospy.get_param("btn_pres", "joy_buttons")
        
        # controller is not straight zero for the axis.. prevent tiny robot movements
        self.deadzone = rospy.get_param("deadzone", 0.05)
        self.joy_pub = rospy.Publisher(self.joy_pub_topic, Joy, queue_size = 1)
        self.joy_btn_pub = rospy.Publisher(self.btn_pres_topic, JoyButtons, queue_size = 1)
        self.maskR = 0xFF0000
        self.maskG = 0x00FF00
        self.maskB = 0x0000FF

        # Subscribers
        self.set_LED_sub = rospy.Subscriber('ds/set_LED', SetColour, self.set_LED, queue_size=10)
        self.set_motor_sub = rospy.Subscriber('ds/set_motor', SetMotor, self.set_motor, queue_size=10)

        self.leak_detected = False
        self.leak_sub = rospy.Subscriber('core/leak', Leak, self.leak_callback, queue_size=10)

    def leak_callback(self, msg: Leak):
        if msg.value and not self.leak_detected:

            self.leak_detected = True

            # Turn on LED
            self.dualsense.setLeftMotor(255)
            self.dualsense.setRightMotor(255)

            # Flash LED red
            while self.leak_detected:

                self.dualsense.light.setColorI(255, 0, 0)
                rospy.sleep(0.3)
                self.dualsense.light.setColorI(0, 0, 0)
                rospy.sleep(0.3)
                
    # ================================================================================
    # DS5 Control
    # ================================================================================

    def set_motor(self, msg: SetMotor):
        self.dualsense.setLeftMotor(msg.left_motor)
        self.dualsense.setRightMotor(msg.right_motor)

    def set_LED(self, msg: SetColour):
        self.dualsense.light.setColorI(msg.R, msg.G, msg.B)
        
    # ================================================================================
    # DS5 Messages
    # ================================================================================
    
    def publish_button_pressed(self):
        
        joy = JoyButtons()
        
        # dpad
        joy.dpad_down = self.dualsense.state.DpadDown == 1
        joy.dpad_up = self.dualsense.state.DpadUp == 1
        joy.dpad_left = self.dualsense.state.DpadLeft == 1
        joy.dpad_right = self.dualsense.state.DpadRight == 1
        
        # Face
        joy.face_circle = self.dualsense.state.circle == 1
        joy.face_square = self.dualsense.state.square == 1
        joy.face_triangle = self.dualsense.state.triangle == 1
        joy.face_x = self.dualsense.state.cross == 1
        
        # Shoulders/Bumpers/Triggers
        joy.shoulder_l1 = self.dualsense.state.L1 == 1
        joy.shoulder_l2 = self.dualsense.state.L2Btn
        joy.shoulder_r1 = self.dualsense.state.R1 == 1
        joy.shoulder_r2 = self.dualsense.state.R2Btn
        
        self.joy_btn_pub.publish(joy)
   
    def joy_publish(self):
        # try:
        joy_msg = Joy()
        #print("in joy_publish")
        # rospy.loginfo('Publishing joy')
        for i in range(18):
            joy_msg.buttons.append(0)
        # Buttons mapping
        joy_msg.buttons[0] = self.dualsense.state.cross
        joy_msg.buttons[1] = self.dualsense.state.circle
        joy_msg.buttons[2] = self.dualsense.state.triangle
        joy_msg.buttons[3] = self.dualsense.state.square
        joy_msg.buttons[13] = self.dualsense.state.DpadUp
        joy_msg.buttons[16] = self.dualsense.state.DpadRight
        joy_msg.buttons[14] = self.dualsense.state.DpadDown
        joy_msg.buttons[15] = self.dualsense.state.DpadLeft
        joy_msg.buttons[6] = self.dualsense.state.L2Btn
        joy_msg.buttons[7] = self.dualsense.state.R2Btn
        joy_msg.buttons[4] = self.dualsense.state.L1
        joy_msg.buttons[5] = self.dualsense.state.R1
        joy_msg.buttons[8] = self.dualsense.state.L3
        joy_msg.buttons[9] = self.dualsense.state.R3
        joy_msg.buttons[10] = self.dualsense.state.ps
        joy_msg.buttons[11] = self.dualsense.state.share
        joy_msg.buttons[12] = self.dualsense.state.options
        joy_msg.buttons[17] = self.dualsense.state.touchBtn
        # Axes mapping
        for i in range(6):
            joy_msg.axes.append(0)
        # change value by Axes from 0.0 -> 1.0
        joy_msg.axes[0] = (-1 * self.dualsense.state.LX )/128.0         #Leftward   (-1.0 -> 1.0, default ~0.0)
        joy_msg.axes[1] = (-1 * self.dualsense.state.LY )/128.0         #Upward     (-1.0 -> 1.0, default ~0.0)
        joy_msg.axes[2] = (-1 * self.dualsense.state.RX )/128.0         #Leftward   (-1.0 -> 1.0, default ~0.0)
        joy_msg.axes[3] = (-1 * self.dualsense.state.RY )/128.0         #Upward     (-1.0 -> 1.0, default ~0.0)
        joy_msg.axes[4] = self.dualsense.state.L2 /255.0                #PushDown   (0.0 -> 1.0, default = 0)
        joy_msg.axes[5] = self.dualsense.state.R2 /255.0                #PushDown   (0.0 -> 1.0, default = 0)
        for val in range(6):
            if(abs(joy_msg.axes[val]) < self.deadzone):
                joy_msg.axes[val] = 0.0
        self.joy_pub.publish(joy_msg)
        # except:
        #     print("catch error")
    def main_loop(self):
        rate = rospy.Rate(self.noderate)
        ### initialize controller if possible, else wait
        # find device and initialize
        while not rospy.is_shutdown():
            if self.node_state == 0:
                try:
                    self.dualsense.init()
                except:
                    rospy.logwarn_throttle(2, "Cannot initialize controller!")
                else:
                    # set rgb led to green
                    self.dualsense.light.setColorI(0,0,255)
                    rospy.loginfo('Connected to controller.')
                    self.node_state = 1
            elif self.node_state == 1:
                # if error -> node_state = 0
                try:
                    self.joy_publish()
                    # self.node_state = 0
                except Exception as e:
                    rospy.logerr(e)
                    rospy.logerr("Lost connection! Go back to init!")
                    self.node_state = 0
                    pass
            rate.sleep()
        self.dualsense.close()
if __name__== '__main__':
    rospy.init_node("ds5ros_node")
    ds5 = Ds5Ros()
    ds5.main_loop()