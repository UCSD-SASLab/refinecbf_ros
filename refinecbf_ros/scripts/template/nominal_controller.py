#!/usr/bin/env python3

import rospy
import numpy as np
from refinecbf_ros.msg import Array
from std_msgs.msg import Bool


class NominalController:
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
        external_control_topic = rospy.get_param("~topics/external_control")
        nominal_control_topic = rospy.get_param("~topics/nominal_control")
        publish_ext_control_flag_topic = rospy.get_param("~topics/publish_external_control_flag")

        # Initialize subscribers and publishers
        self.state_sub = rospy.Subscriber(state_topic, Array, self.callback_state)
        self.control_pub = rospy.Publisher(nominal_control_topic, Array, queue_size=1)
        self.external_control_sub = rospy.Subscriber(external_control_topic, Array, self.callback_external_control)
        self.publish_ext_control_flag_pub = rospy.Publisher(publish_ext_control_flag_topic, Bool, queue_size=1)

        # Initialize control variables
        self.external_control = None
        self.new_external_control = False

        # Initialize Controller
        self.controller = None

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

    def prioritize_control(self, control):
        """
        Prioritizes the external control if the override_nominal_control() conditions are met (robot specific).

        Args:
            control (Array): The nominal control.

        Returns:
            Array: The prioritized control.

        """
        if self.new_external_control:
            self.publish_ext_control_flag_pub.publish(True)
            self.new_external_control = False
            return self.external_control
        else:
            self.publish_ext_control_flag_pub.publish(False)
            return control


    def publish_control(self):
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
