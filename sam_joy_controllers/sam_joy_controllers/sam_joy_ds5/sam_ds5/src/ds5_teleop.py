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

# Create a ROS node that republishes a joy message


import rospy

from sensor_msgs.msg import JoyFeedbackArray, Joy
from smarc_msgs.msg import ThrusterRPM
from sam_msgs.msg import ThrusterAngles

from ds5_msgs.msg import SetColour
from ds5_msgs.msg import SetMotor

from std_msgs.msg import Bool

import threading
import time

class ds5_teleop():
    
    # ================================================================================
    # LED Feedback Functions
    # ================================================================================
    
    def set_LED(self, R: int, G: int, B: int):
        """
        Set the LED colour on the DS5 controller
        
        Parameters
        ----------
        R : int
            Red value between 0 and 255
        G : int
            Green value between 0 and 255
        B : int
            Blue value between 0 and 255
        """
        
        colour_msg = SetColour()
        colour_msg.R = R
        colour_msg.G = G
        colour_msg.B = B
        self.setLED.publish(colour_msg)
        rospy.loginfo("Set LED colour to R: {}, G: {}, B: {}".format(R, G, B))
    
    # ================================================================================
    # Motor Functions
    # ================================================================================
    
    def set_motor(self, left_motor: int, right_motor: int):
        """
        Set the motor speed on the DS5 controller
        
        Parameters
        ----------
        left_motor : int
            Left motor speed between 0 and 255
        right_motor : int
            Right motor speed between 0 and 255
        """
        
        motor_msg = SetMotor()
        motor_msg.left_motor = left_motor
        motor_msg.right_motor = right_motor
        self.setMotor.publish(motor_msg)
    
    def send_motor_pulse(self, pulse_length: float, no_pulses = 1):
        """
        Send a short motor pulse to the DS5 controller. 
        
        Each time the function is called, a new thread is created to send the pulse. 
        This prevents the function from blocking the main thread.
        
        Parameters
        ----------
        pulse_length : float
            Length of the pulse in seconds, should be a multiple of 0.1 seconds
        no_pulses : int
            Number of pulses to send
        """
        
        def send_pulse_task(no_pulses):
            last_press = not self.button_pressed_flag.is_set()
            self.button_pressed_flag.set()
            for i in range(no_pulses):            
                self.set_motor(255, 255)   
                # Sleep for pulse length but check if the thread has been killed
                for i in range(int(pulse_length / 0.1)):
                    # Ensure that only the latest thread runs to completion
                    if self.button_pressed_flag.is_set() or last_press:
                        self.button_pressed_flag.clear()
                        time.sleep(0.05)
                    else:
                        return
                self.set_motor(0, 0)   
                # Sleep for pulse length but check if the thread has been killed
                for i in range(int(pulse_length / 0.1)):
                    # Ensure that only the latest thread runs to completion
                    if self.button_pressed_flag.is_set() or last_press:
                        self.button_pressed_flag.clear()
                        time.sleep(0.05)
                    else:
                        return       
                
        thread = threading.Thread(target=send_pulse_task(no_pulses))
        thread.start()
            
    # ================================================================================
    # Callback Functions
    # ================================================================================

    def joy_callback(self, msg:Joy):
        """
        Callback function for the joystick subscriber
        """

        x_pressed = msg.buttons[0] == 1

        if x_pressed and not self.enable_teleop_pressed:
            self.teleop_enabled = not self.teleop_enabled
            self.enable_teleop_pressed = True
            rospy.loginfo("Teleop enabled: {}".format(self.teleop_enabled))

            # Set LED colour
            self.set_LED(255 if not self.teleop_enabled else 0, 255 if self.teleop_enabled else 0, 0)
            
            if self.teleop_enabled:
                self.send_motor_pulse(0.2, 1)
            else:
                self.send_motor_pulse(0.2, 2)

        if not x_pressed:
            self.enable_teleop_pressed = False

        if self.teleop_enabled:

            RPM_MAX = 1500
            RAD_MAX = 0.1
            RAD_STEPS = 5
            RAD_STEP_SIZE = RAD_MAX / RAD_STEPS

            rpm_cmd = int(msg.axes[1] * RPM_MAX)
            x_cmd = msg.axes[2] * RAD_MAX
            y_cmd = msg.axes[3] * RAD_MAX

            # Round rpm_cmd to nearest 100
            rpm_cmd = int(round(rpm_cmd, -2))

            # Round x_cmd and y_cmd to nearest value on range RAD_MIN to RAD_MAX with RAD_STEPS steps
            x_steps = round(x_cmd / RAD_STEP_SIZE)
            x_cmd = x_steps * RAD_STEP_SIZE
            y_steps = round(y_cmd / RAD_STEP_SIZE)
            y_cmd = y_steps * RAD_STEP_SIZE

            rpm1_msg = ThrusterRPM()
            rpm2_msg = ThrusterRPM()
            rpm1_msg.rpm = rpm_cmd
            rpm2_msg.rpm = rpm_cmd

            angle_msg = ThrusterAngles()
            angle_msg.thruster_vertical_radians = y_cmd
            angle_msg.thruster_horizontal_radians = x_cmd*-1

            self.rpm1_pub.publish(rpm1_msg)
            self.rpm2_pub.publish(rpm2_msg)
            self.angle_pub.publish(angle_msg)
            
    def send_teleop_enabled(self, teleop_enabled: bool):
        """
        Send a teleop enabled message to the core
        """
        
        teleop_enabled_msg = Bool()
        teleop_enabled_msg.data = teleop_enabled
        self.teleop_enabled_pub.publish(teleop_enabled_msg)


    # ================================================================================
    # Node Init and shutdown
    # ================================================================================
    
    def shutdown(self):
        
        rospy.loginfo("Shutting down DS5 teleop node")
        
        # On shutdown, send a motor pulse to indicate shutdown and send teleop disabled
        self.set_LED(0, 0, 255)
        self.send_teleop_enabled(False)
        self.send_motor_pulse(0.2, 2)

    def __init__(self):
        
        # Threading flags
        self.button_pressed_flag = threading.Event()

        # pub = rospy.Publisher('chatter', String, queue_size=10)
        rospy.init_node('ds5_teleop', anonymous=True)
        rate = rospy.Rate(1) # 10hz

        # Publishers
        self.rpm1_pub = rospy.Publisher('core/thruster1_cmd', ThrusterRPM, queue_size=2)
        self.rpm2_pub = rospy.Publisher('core/thruster2_cmd', ThrusterRPM, queue_size=2)
        self.angle_pub = rospy.Publisher('core/thrust_vector_cmd', ThrusterAngles, queue_size=2) 
        self.teleop_enabled_pub = rospy.Publisher('core/teleop/enable', Bool, queue_size=10)

        # DS5 publishers
        self.setLED = rospy.Publisher('ds/set_LED', SetColour, queue_size=10)
        self.setMotor = rospy.Publisher('ds/set_motor', SetMotor, queue_size=10)

        # Keep track of buttons pressed
        self.enable_teleop_pressed = False

        # States
        self.teleop_enabled = False

        # Joy subscriber
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback)
                
        rospy.on_shutdown(self.shutdown)

        while not rospy.is_shutdown():
            self.send_teleop_enabled(self.teleop_enabled)
            rate.sleep()        

if __name__ == '__main__':
    try:
        ds5_teleop()
    except rospy.ROSInterruptException:
        pass