#!/usr/bin/env python3

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from refinecbf_ros.msg import StateArray, ControlArray
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

    def callback_state(self, state_in_msg):
        w = state_in_msg.pose.pose.orientation.w
        x = state_in_msg.pose.pose.orientation.x
        y = state_in_msg.pose.pose.orientation.y
        z = state_in_msg.pose.pose.orientation.z

        # Convert Quaternion to Yaw
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (np.power(y, 2) + np.power(z, 2)))

        state_out_msg = StateArray()
        state_out_msg.state = [state_in_msg.pose.pose.position.x, state_in_msg.pose.pose.position.y, yaw]
        self.state_pub.publish(state_out_msg)

    def callback_safe_control(self, control_in_msg):
        control_in = control_in_msg.control
        control_out_msg = self.safe_control_msg_type()
        control_out_msg.linear.x = control_in[1]
        control_out_msg.linear.y = 0.0
        control_out_msg.linear.z = 0.0

        control_out_msg.angular.x = 0.0
        control_out_msg.angular.y = 0.0
        control_out_msg.angular.z = control_in[0]
        self.safe_control_pub.publish(control_out_msg)

    def callback_external_control(self, control_in_msg):
        # When nominal control comes through the HW interface, it is a Twist message
        control_out_msg = ControlArray()
        control_out_msg.control = [control_in_msg.angular.z, control_in_msg.linear.x]
        self.external_control_pub.publish(control_out_msg)


if __name__ == "__main__":
    rospy.init_node("interface")
    TurtlebotInterface()
    rospy.spin()
