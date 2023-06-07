#!/usr/bin/python3

import rospy
from rospy import ROSException
from std_msgs.msg import Header, Bool
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion
from sensor_msgs.msg import NavSatFix, NavSatStatus
from sam_msgs.msg import GetGPSFixAction, GetGPSFixFeedback, GetGPSFixResult
from sam_msgs.msg import PercentStamped 
import actionlib
import tf_conversions
import tf
from tf.transformations import quaternion_from_euler, quaternion_multiply
from geodesy import utm
import math
import numpy as np

class GPSFixServer(object):

    _feedback = GetGPSFixFeedback()
    _result = GetGPSFixResult()

    def __init__(self, name):

        self.last_gps_pos = None
        self.last_dr_pos = None

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, GetGPSFixAction, execute_cb=self.execute_cb, auto_start=False)

        self.pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.lcg_disable_pub = rospy.Publisher('/sam/ctrl/lcg/pid_enable', Bool, queue_size=10)
        self.vbs_disable_pub = rospy.Publisher('/sam/ctrl/vbs/pid_enable', Bool, queue_size=10)

        self.lcg_pub = rospy.Publisher('/sam/core/lcg_cmd', PercentStamped, queue_size=10)
        self.vbs_pub = rospy.Publisher('/sam/core/vbs_cmd', PercentStamped, queue_size=10)

        self.listener = tf.TransformListener()

        self._as.start()

    def start_stop_dvl(self, value, value_string):
        try:
            rospy.wait_for_service('/sam/core/start_stop_dvl', timeout=3.)
            start_stop_dvl = rospy.ServiceProxy('/sam/core/start_stop_dvl', SetBool)
            resp = start_stop_dvl(value)
            if not resp.success:
                self._feedback.status = "Service call returned false, failed to %s dvl" % value_string
                rospy.loginfo("Service call returned false, failed to %s dvl", value_string)
        except (rospy.ServiceException, ROSException) as e:
            self._feedback.status = "Service call failed, failed to %s dvl" % value_string
            rospy.loginfo("Service call failed: %s, failed to %s dvl", e, value_string)
        #finally:
        #    self._feedback.status = "Did %s dvl" % (value_string)
        self._as.publish_feedback(self._feedback)

    def estimate_position(self, fixes, covars):
            
        try:
            now = rospy.Time(0)
            (world_trans, world_rot) = self.listener.lookupTransform("world_utm", "world_local", now)
        except (tf.LookupException, tf.ConnectivityException):
            self._feedback.status = "Could not get transform between %s and %s" % ("world_utm", "world_local")
            rospy.loginfo("Could not get transform between %s and %s" % ("world_utm", "world_local"))
            self._as.publish_feedback(self._feedback)
            
        # easting, northing is in world_utm coordinate system,
        # we need to transform it to world or world_local
        
        pos = np.zeros((len(fixes), 3))
        for i, fix in enumerate(fixes):
            utm_point = utm.fromLatLong(fix[0], fix[1])
            easting = utm_point.easting
            northing = utm_point.northing
            utm_zone = utm_point.zone

            pos[i, :] = np.array([easting-world_trans[0], northing-world_trans[1], 0.])

        # use the cov to weight the means in the future
        estimate = np.mean(pos, axis=0)

        return estimate

    def execute_cb(self, goal):

        rospy.loginfo("Got action callback...")
        self._feedback.status = "Shutting down controllers and DVL"
        self._as.publish_feedback(self._feedback)

        header = Header()
        timeout = goal.timeout
        required_gps_msgs = goal.required_gps_msgs

        self.start_stop_dvl(False, "stop")

        # Disable controllers
        self.vbs_disable_pub.publish(False)
        self.lcg_disable_pub.publish(False)

        # Sleep to make sure controllers are down
        rospy.sleep(0.1)

        # Set VBS to 0
        self.vbs_pub.publish(0., header)

        # Set LCG to 0
        self.lcg_pub.publish(0., header)

        good_fixes = []
        good_vars = [] # NOTE: covariances are in m^2

        # Get GPS fixes until we are in a good place
        gps_topic = "/sam/core/gps"
        start_time = rospy.get_time()
        while rospy.get_time() - start_time < timeout and len(good_fixes) < required_gps_msgs:
            try:
                gps_msg = rospy.wait_for_message(gps_topic, NavSatFix, 3.)
            except rospy.ROSException:
                rospy.loginfo("Could not get gps message on %s, aborting...", gps_topic)
                self._feedback.status = "Could not get gps message on %s..." % gps_topic
                self._as.publish_feedback(self._feedback)
                continue
            if gps_msg.status.status != NavSatStatus.STATUS_NO_FIX:
                self._feedback.status = "Good fix, now has %d msgs" % len(good_fixes)
                good_fixes.append(np.array([gps_msg.latitude, gps_msg.longitude]))
                good_vars.append(np.array([gps_msg.position_covariance[:2], gps_msg.position_covariance[3:5]]))
            else:
                self._feedback.status = "No fix, now has %d msgs" % len(good_fixes)
            self._as.publish_feedback(self._feedback)

        if len(good_fixes) < required_gps_msgs:
            self._result.status = "Timeout, not enough msgs"
            self._as.set_aborted(self._result)
            return
        else:
            self._feedback.status = "Done listening, got %d msgs" % len(good_fixes)
            self._as.publish_feedback(self._feedback)

        self.start_stop_dvl(True, "start")
        gps_pos = self.estimate_position(good_fixes, good_vars)

        corrected_rot = [0., 0., 0., 1.] # Start with 0 yaw

        if self.last_dr_pos is not None and self.last_gps_pos is not None:
            self._feedback.status = "Found previous positions, doing heading estimation"
            self._as.publish_feedback(self._feedback)

            try:
                now = rospy.Time(0)
                (dr_trans, dr_rot) = self.listener.lookupTransform("world_local", "sam/base_link", now)
            except (tf.LookupException, tf.ConnectivityException):
                self._feedback.status = "Could not get transform between %s and %s" % ("world_local", "sam/base_link")
                rospy.loginfo("Could not get transform between %s and %s" % ("world_local", "sam/base_link"))
                self._as.publish_feedback(self._feedback)
            rospy.sleep(0.3)

            gps_diff =  gps_pos - self.last_gps_pos
            #gps_diff = 1./np.linalg.norm(gps_diff)*gps_diff
            gps_trajectory_yaw = math.atan2(gps_diff[1], gps_diff[0])

            dr_diff = np.array((dr_trans[0] - self.last_dr_pos[0], dr_trans[1] - self.last_dr_pos[1]))
            #dr_diff = 1./np.linalg.norm(dr_diff)*dr_diff
            dr_trajectory_yaw = math.atan2(dr_diff[1], dr_diff[0])

            yaw_correction = gps_trajectory_yaw - dr_trajectory_yaw
            # to get the actual yaw, we need to look at the
            # the difference in odom between last time and this time
            # note that we need to get the new estimated yaw
            # after publishing this to get the corrected one

            self._feedback.status = "Estimated GPS yaw: %f, DR yaw: %f, Yaw corr: %f" % (gps_trajectory_yaw, dr_trajectory_yaw, yaw_correction)
            self._as.publish_feedback(self._feedback)
            rospy.sleep(0.3)

            corrected_rot = quaternion_multiply(quaternion_from_euler(0., 0., yaw_correction), dr_rot)

        self._feedback.status = "Waiting for filter to update"
        self._as.publish_feedback(self._feedback)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "world_local"
        pose_msg.pose.pose.position = Point(*gps_pos.tolist())
        pose_msg.pose.pose.orientation = Quaternion(*corrected_rot)
        self.pose_pub.publish(pose_msg)
        rospy.sleep(.5)

        self._feedback.status = "Getting updated pose"
        self._as.publish_feedback(self._feedback)
        try:
            now = rospy.Time(0)
            (trans, rot) = self.listener.lookupTransform("world_local", "sam/base_link", now)
            self.last_dr_pos = trans
        except (tf.LookupException, tf.ConnectivityException):
            self._feedback.status = "Could not get transform between %s and %s" % ("world_local", "sam/base_link")
            rospy.loginfo("Could not get transform between %s and %s" % ("world_local", "sam/base_link"))
            self._as.publish_feedback(self._feedback)
        rospy.sleep(0.3)

        self.last_gps_pos = gps_pos

        self._result.status = "Finished setting position"
        self._as.set_succeeded(self._result)

if __name__ == "__main__":

    rospy.init_node('gps_fix_server', anonymous=False) #True)

    check_server = GPSFixServer(rospy.get_name())

    rospy.spin()
