#!/usr/bin/env python3

import rospy
import numpy as np
from refinecbf_ros.msg import Array
from std_msgs.msg import Bool


class Disturbance:
    """
    This class determines the non-safe (aka nominal) control of a robot using hj_reachability in refineCBF.
    External control inputs (keyboard, joystick, etc.) by default override the nominal control from the autonomy stack if they have been published and/or changed recently.

    Subscribers:
    - state_sub (~topics/state): Subscribes to the robot's state.
    - external_control_sub (~topics/external_control): Subscribes to the external control input.

    Publishers:
    - control_pub (~topics/nominal_control): Publishes the nominal control.

    """

    def __init__(self):
        """
        Initialize the NominalController.
        """
        # Get topics from parameters
        state_topic = rospy.get_param("~topics/state")
        publish_disturbance_flag_topic = rospy.get_param("~topics/publish_disturbance_flag")
        disturbance_topic = rospy.get_param("~topics/disturbance")
        # Initialize subscribers and publishers
        self.state_sub = rospy.Subscriber(state_topic, Array, self.callback_state)
        self.disturbance_pub = rospy.Publisher(disturbance_topic, Array, queue_size=1)

        # Initialize Controller
        self.disturbance = None

    def callback_state(self, state_array_msg):
        """
        Callback for the state subscriber.

        Args:
            state_array_msg (Array): The incoming state message.

        This method updates the robot's state based on the incoming message.
        """
        self.state = np.array(state_array_msg.value)

    def callback_external_control(self, control_msg):
        """
        Callback for the external control subscriber.

        Args:
            control_msg (Array): The incoming control message.

        This method updates the external control based on the incoming message.
        Sets the new_external_control flag to True.
        """
        self.external_control = np.array(control_msg.value)
        self.new_external_control = True


    def publish_disturbance(self):
        """
        Publishes the prioritized control.

        This method gets the nominal control, prioritizes it with the external control input,
        and publishes the result.
        """
        # Get nominal control
        control = self.controller(self.state, rospy.get_time()).squeeze()

        # Prioritize between external input (e.g. joystick) and nominal control
        control = self.prioritize_control(control)

        # Create control message
        control_msg = Array()
        control_msg.value = control.tolist()

        # Publish control message
        self.control_pub.publish(control_msg)


# Idea: The disturbance controller should be able to receive a flag on whether it can be active (e.g. only after takeoff)
# Next: It should have access to a randomizer, which can sample from the disturbance space
# Next: It should have access to the disturbance model and based on that output the disturbance in the state space (based on the state space of the model)
# Next: It should publish this to the disturbance topic and then in the cf_hw_interface, the disturbance needs to be converted