#!/usr/bin/env python3

import rospy
import numpy as np

from refinecbf_ros.msg import StateArray, ControlArray


class BaseInterface:
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the crazyflies.
    Each HW platform should have its own Interface node
    """

    state_msg_type = None
    safe_control_msg_type = None
    external_control_msg_type = None

    def __init__(self):
        # Convert state messages to correct type
        robot_state_topic = rospy.get_param("~topics/robot_state", "/state")
        cbf_state_topic = rospy.get_param("~topics/cbf_state", "/state_array")

        state_sub = rospy.Subscriber(robot_state_topic, self.state_msg_type, self.callback_state)
        self.state_pub = rospy.Publisher(cbf_state_topic, StateArray, queue_size=1)

        # Convert control messages to correct type
        robot_safe_control_topic = rospy.get_param("~topics/robot_safe_control")
        robot_external_control_topic = rospy.get_param("~topics/robot_external_control")

        cbf_safe_control_topic = rospy.get_param("~topics/cbf_safe_control")
        cbf_external_control_topic = rospy.get_param("~topics/cbf_external_control")

        safe_control_sub = rospy.Subscriber(cbf_safe_control_topic, ControlArray, self.callback_safe_control)
        # PrioritizedControlStamped to give a "priority value" to the control (used in the merged_control node)
        self.safe_control_pub = rospy.Publisher(robot_safe_control_topic, self.safe_control_msg_type, queue_size=1)

        external_control_sub = rospy.Subscriber(
            robot_external_control_topic, self.external_control_msg_type, self.callback_external_control
        )
        self.external_control_pub = rospy.Publisher(cbf_external_control_topic, ControlArray, queue_size=1)

    def callback_state(self, state_msg):
        raise NotImplementedError("Must be subclassed")

    def callback_safe_control(self, control_msg):
        raise NotImplementedError("Must be subclassed")

    def callback_external_control(self, control_msg):
        raise NotImplementedError("Must be subclassed")
