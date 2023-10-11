#!/usr/bin/env python3

import rospy
import numpy as np

from crazyflie_msgs.msg import PositionVelocityYawStateStamped, PrioritizedControlStamped, ControlStamped
from refinecbf_ros.msg import StateArray, ControlArray

# add current folder to pythonpath FIXME Is this necessary
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

    def callback_state(self, state_msg):
        #  state_msg is a PositionVelocityYawStateStamped message by default, we only care about planar motion
        state_array = StateArray()
        state_array.state = [state_msg.state.y, state_msg.state.z, state_msg.state.y_dot, state_msg.state.z_dot]
        self.state_pub.publish(state_array)

    def callback_safe_control(self, control_msg):
        # control_msg is a ControlArray message of size 2 (np.tan(roll), thrust) for the current dynamics,
        # we convert it back to size 4 (roll, pitch, yaw_dot, thrust) and publish
        control = control_msg.control
        current_control = PrioritizedControlStamped()
        current_control.header.stamp = rospy.Time.now()
        current_control.control.priority = 1.0
        current_control.control.control.roll = np.arctan(control[0])
        current_control.control.control.pitch = 0.0
        current_control.control.control.yaw_dot = 0.0
        current_control.control.control.thrust = control[1]
        self.safe_control_pub.publish(current_control)

    def callback_external_control(self, control_msg):
        # Inverse operation from callback_safe_control
        control = control_msg.control
        current_control = ControlArray()
        current_control.control = [np.tan(control.roll), control.thrust]
        self.external_control_pub.publish(current_control)


if __name__ == "__main__":
    rospy.init_node("interface")
    CrazyflieInterface()
    rospy.spin()
