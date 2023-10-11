#!/usr/bin/env python3

import rospy
import numpy as np

from crazyflie_msgs.msg import PositionVelocityYawStateStamped, PrioritizedControlStamped, ControlStamped
from refinecbf_ros.msg import StateArray, ControlArray
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from template.hw_interface import BaseInterface


class CrazyflieInterface(BaseInterface):
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the crazyflies.
    Each HW platform should have its own Interface node
    """

    state_msg_type = PositionVelocityYawStateStamped
    safe_control_msg_type = PrioritizedControlStamped
    external_control_msg_type = ControlStamped

    def callback_state(self, state_in_msg):
        #  state_msg is a PositionVelocityYawStateStamped message by default, we only care about planar motion
        state_out_msg = StateArray()
        state_out_msg.state = [
            state_in_msg.state.y,
            state_in_msg.state.z,
            state_in_msg.state.y_dot,
            state_in_msg.state.z_dot,
        ]
        self.state_pub.publish(state_out_msg)

    def callback_safe_control(self, control_in_msg):
        # control_msg is a ControlArray message of size 2 (np.tan(roll), thrust) for the current dynamics,
        # we convert it back to size 4 (roll, pitch, yaw_dot, thrust) and publish
        control_in = control_in_msg.control
        control_out_msg = self.safe_control_msg_type()
        control_out_msg.header.stamp = rospy.Time.now()
        control_out_msg.control.priority = 1.0
        control_out_msg.control.control.roll = np.arctan(control_in[0])
        control_out_msg.control.control.pitch = 0.0
        control_out_msg.control.control.yaw_dot = 0.0
        control_out_msg.control.control.thrust = control_in[1]
        self.safe_control_pub.publish(control_out_msg)

    def callback_external_control(self, control_in_msg):
        # Inverse operation from callback_safe_control
        control = control_in_msg.control
        control_out_msg = ControlArray()
        control_out_msg.control = [np.tan(control.roll), control.thrust]
        self.external_control_pub.publish(control_out_msg)


if __name__ == "__main__":
    rospy.init_node("interface")
    CrazyflieInterface()
    rospy.spin()
