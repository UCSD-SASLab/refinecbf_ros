#!/usr/bin/env python3

import rospy
import numpy as np

from crazyflie_msgs.msg import PositionVelocityYawStateStamped, PrioritizedControlStamped, ControlStamped
from refinecbf_ros.msg import StateArray, ControlArray

class CrazyflieInterface:
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the crazyflies. 
    Each HW platform should have its own Interface node
    """

    def __init__(self):
        # Convert state messages to correct type
        robot_state_topic = rospy.get_param("~topics/robot_state", "/state")
        cbf_state_topic = rospy.get_param("~topics/cbf_state", "/state_array")

        state_sub = rospy.Subscriber(robot_state_topic, PositionVelocityYawStateStamped, self.callback_state)
        self.state_pub = rospy.Publisher(cbf_state_topic, StateArray, queue_size=1)

        # Convert control messages to correct type
        robot_safe_control_topic = rospy.get_param("~topics/robot_safe_control")
        robot_nominal_control_topic = rospy.get_param("~topics/robot_nominal_control")

        cbf_safe_control_topic = rospy.get_param("~topics/cbf_safe_control")
        cbf_nominal_control_topic = rospy.get_param("~topics/cbf_nominal_control")

        safe_control_sub = rospy.Subscriber(cbf_safe_control_topic, ControlArray, self.callback_safe_control)
        # PrioritizedControlStamped to give a "priority value" to the control (used in the merged_control node)
        self.safe_control_pub = rospy.Publisher(robot_safe_control_topic, PrioritizedControlStamped, queue_size=1)

        nominal_control_sub = rospy.Subscriber(robot_nominal_control_topic, ControlStamped, self.callback_nominal_control)
        self.nominal_control_pub = rospy.Publisher(cbf_nominal_control_topic, ControlArray, queue_size=1)

    def callback_state(self, state_msg):
        #  state_msg is a PositionVelocityYawStateStamped message by default, we only care about planar motion
        state_array = StateArray()
        state_array.state = [state_msg.state.y, state_msg.state.z, state_msg.state.y_dot, state_msg.state.z_dot]
        self.state_pub.publish(state_array)

    def callback_safe_control(self, control_msg):
        # control_msg is a ControlArray message of size 2 (np.tan(roll), thrust) forr the current dynamics, 
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

    def callback_nominal_control(self, control_msg):
        # Inverse operation from callback_safe_control
        control = control_msg.control
        current_control = ControlArray()
        current_control.control = [np.tan(control.roll), control.thrust]
        self.nominal_control_pub.publish(current_control)



if __name__ == "__main__":
    rospy.init_node("interface")
    CrazyflieInterface()
    rospy.spin()