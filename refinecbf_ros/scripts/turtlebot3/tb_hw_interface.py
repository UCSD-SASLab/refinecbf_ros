#!/usr/bin/env python3

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from refinecbf_ros.msg import StateArray, ControlArray
# add current folder to pythonpath FIXME Is this necessary
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from template.hw_interface import BaseInterface


class TurtlebotInterface(BaseInterface):
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the Turtlebots. 
    Each HW platform should have its own Interface node
    """
    state_msg_type = Odometry
    safe_control_msg_type = Twist
    external_control_msg_type = Twist

    def callback_state(self,odom_msg):

        w = odom_msg.pose.pose.orientation.w
        x = odom_msg.pose.pose.orientation.x
        y = odom_msg.pose.pose.orientation.y
        z = odom_msg.pose.pose.orientation.z

        # Convert Quaternion to Yaw
        yaw = np.arctan2(2*(w*z+x*y),1-2*(np.power(y,2)+np.power(z,2)))

        state_array_msg = StateArray()
        state_array_msg.state = [odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y,yaw]
        self.state_pub.publish(state_array_msg)

    def callback_safe_control(self, control_msg):
        control = control_msg.control
        safe_control_out_msg = Twist()
        safe_control_out_msg.linear.x = control[1]
        safe_control_out_msg.linear.y = 0.0
        safe_control_out_msg.linear.z = 0.0

        safe_control_out_msg.angular.x = 0.0
        safe_control_out_msg.angular.y = 0.0
        safe_control_out_msg.angular.z = control[0]
        self.safe_control_pub.publish(safe_control_out_msg)

    def callback_external_control(self, control_msg):
        # When nominal control comes through the HW interface, it is a Twist message
        nominal_control_out_msg = ControlArray()
        nominal_control_out_msg.control = [control_msg.angular.z, control_msg.linear.x]
        self.external_control_pub.publish(nominal_control_out_msg)



if __name__ == "__main__":
    rospy.init_node("interface")
    TurtlebotInterface()
    rospy.spin()
