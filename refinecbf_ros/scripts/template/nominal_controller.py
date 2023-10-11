#!/usr/bin/env python3

import rospy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp

from refinecbf_ros.msg import StateArray, ControlArray
from refinecbf_ros.config import Config
from time import time


class NominalController:
    """
    This class determines the nominal control for any robotic system. By default it
    """

    def __init__(self):
        # Subscribers:
        #   StateArray - Turtlebot State
        # Publishers:
        #   ControlArray - Turtlebot Control

        # Topics
        state_topic = rospy.get_param("~topics/state")
        external_control_topic = rospy.get_param("~topics/external_control")
        nominal_control_topic = rospy.get_param("~topics/nominal_control")

        self.state_sub = rospy.Subscriber(state_topic, StateArray, self.callback_state)
        self.control_pub = rospy.Publisher(nominal_control_topic, ControlArray, queue_size=1)
        self.external_control_sub = rospy.Subscriber(
            external_control_topic, ControlArray, self.callback_external_control
        )

        self.external_control_buffer = rospy.get_param("/ctr/external_control_buffer", 1.0)  # seconds
        self.external_control_ts = None
        # Initialize Controller
        self.controller = None

    def callback_state(self, state_array_msg):
        self.state = np.array(state_array_msg.state)

    def callback_external_control(self, control_msg):
        self.external_control_ts = rospy.get_time()
        self.external_control = np.array([control_msg.control])

    def prioritize_control(self, control):
        """
        This function prioritizes the external control if it has been published recently
        Could be overriden by child class if desired TODO: Add customizability
        """
        curr_time = rospy.get_time()
        if (
            self.external_control_ts is not None
            and (curr_time - self.external_control_ts).to_sec() < self.external_control_buffer
        ):
            return self.external_control
        else:
            return control

    def publish_control(self):
        control = self.controller.get_nominal_control(self.state, rospy.get_time())[0]
        control = self.prioritize_control(
            control
        )  # Prioritizes between external input (e.g. joystick) and nominal control
        control_msg = ControlArray()
        control_msg.control = control.tolist()
        self.control_pub.publish(control_msg)
