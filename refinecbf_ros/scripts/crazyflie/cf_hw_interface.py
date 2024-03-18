#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Empty
from crazyflie_msgs.msg import (
    PositionVelocityYawStateStamped,
    PrioritizedControlStamped,
    PositionVelocityStateStamped,
    ControlStamped,
    DisturbanceStamped,
)
from refinecbf_ros.msg import Array
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from template.hw_interface import BaseInterface


class CrazyflieInterface(BaseInterface):
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the crazyflies.
    Each HW platform should have its own Interface node
    """

    state_msg_type = PositionVelocityYawStateStamped
    control_out_msg_type = ControlStamped
    external_control_msg_type = ControlStamped
    disturbance_out_msg_type = DisturbanceStamped

    def __init__(self):
        # In flight flag setup
        self.in_flight_flag_topic = rospy.get_param("in_flight_topic", "/control/in_flight")
        self.is_in_flight = False
        
        # External setpoint setup
        self.external_setpoint_time_buffer = rospy.get_param("/ctr/external_setpoint_buffer", 30.0)  # seconds
        self.external_setpoint_ts = None
        self.external_setpoint = None
        
        # Control bounds
        self.max_thrust = rospy.get_param("~control/max_thrust")
        self.min_thrust = rospy.get_param("~control/min_thrust")
        self.max_roll = rospy.get_param("~control/max_roll")
        self.max_pitch = rospy.get_param("~control/max_pitch")

        super().__init__()

    def init_subscribers(self):
        super().init_subscribers()
        self.in_flight_flag_sub = rospy.Subscriber(self.in_flight_flag_topic, Empty, self.callback_in_flight)
        self.external_setpoint_sub = rospy.Subscriber("/control/external_setpoint", PositionVelocityStateStamped, self.callback_setpoint)  # TODO: Remove hardcoding

    def callback_state(self, state_in_msg):
        #  state_msg is a PositionVelocityYawStateStamped message
        state_out_msg = Array()
        state_out_msg.value = [
            state_in_msg.state.x,
            state_in_msg.state.y,
            state_in_msg.state.z,
            state_in_msg.state.x_dot,
            state_in_msg.state.y_dot,
            state_in_msg.state.z_dot,
            state_in_msg.state.yaw
        ]
        self.state_pub.publish(state_out_msg)

    def process_safe_control(self, control_in_msg):
        # control_msg is a ControlArray message of size 2 (np.tan(roll), thrust) for the current dynamics,
        # we convert it back to size 4 (roll, pitch, yaw_dot, thrust) and publish
        control_in = control_in_msg.value
        control_out_msg = self.control_out_msg_type()
        control_out_msg.header.stamp = rospy.Time.now()
        if self.control_out_msg_type == ControlStamped:
            control_out_msg.control.roll = np.arctan(control_in[0])  # TODO: Seems hacky to have np.arctan here
            control_out_msg.control.pitch = control_in[1]
            control_out_msg.control.yaw_dot = control_in[2]
            control_out_msg.control.thrust =  control_in[3]
            control_out_msg.control = self.clip_control_output(control_out_msg.control)
        else:
            raise ValueError("Invalid safe control message type: {}".format(self.control_out_msg_type))
        return control_out_msg

    def clip_control_output(self, control_in_msg):
        control_in_msg.roll = np.minimum(self.max_roll, np.maximum(-self.max_roll, control_in_msg.roll))
        control_in_msg.pitch = np.minimum(self.max_pitch, np.maximum(-self.max_pitch, control_in_msg.pitch))
        control_in_msg.yaw_dot = control_in_msg.yaw_dot
        control_in_msg.thrust = np.minimum(self.max_thrust, np.maximum(self.min_thrust, control_in_msg.thrust))
        return control_in_msg

    def process_external_control(self, control_in_msg):
        # Inverse operation from process_safe_control
        self.external_control_robot = control_in_msg
        control = control_in_msg.control
        control_out_msg = Array()
        control_out_msg.value = [np.tan(control.roll), control.pitch, control.yaw_dot, control.thrust]
        return control_out_msg
    
    def process_disturbance(self, disturbance_in_msg):
        disturbance_in = disturbance_in_msg.value
        disturbance_out_msg = self.disturbance_out_msg_type()
        disturbance_out_msg.header.stamp = rospy.Time.now()
        disturbance_out_msg.disturbance.y = disturbance_in[0]
        disturbance_out_msg.disturbance.z = disturbance_in[1]
        disturbance_out_msg.disturbance.y_dot = disturbance_in[2]
        disturbance_out_msg.disturbance.z_dot = disturbance_in[3]
        return disturbance_out_msg

    def override_safe_control(self):
        return not self.is_in_flight  # If idle / taking off / landing -> override

    def callback_in_flight(self, msg):
        self.is_in_flight = not self.is_in_flight

    def override_nominal_control(self):
        curr_time = rospy.get_time()
        # Prioritize external control only if it exists and when:
        #  - Recent new reference
        #  - TODO: Add clause for joystick external controls
        return (
            self.external_setpoint is not None
            and (curr_time - self.external_setpoint_ts) <= self.external_setpoint_time_buffer
        )

    def callback_setpoint(self, msg):
        self.external_setpoint = True
        rospy.loginfo(
            "Received external setpoint, should prioritize external control for {} seconds".format(
                self.external_setpoint_time_buffer
            )
        )
        self.external_setpoint_ts = rospy.get_time()


if __name__ == "__main__":
    rospy.init_node("interface")
    CrazyflieInterface()
    rospy.spin()
