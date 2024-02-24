#!/usr/bin/env python3

import rospy
import numpy as np
from refinecbf_ros.msg import Array, HiLoArray
from refinecbf_ros.config import Config


class DisturbanceNode:
    """
    This node is responsible for generating disturbances for simulator purposes. Hardware does not have this issue.
    """

    def __init__(self):
        config = Config(hj_setup=False)
        disturbance_topic = rospy.get_param("~topics/disturbance")
        self.pub_disturbance = rospy.Publisher(disturbance_topic, Array, queue_size=1)

        disturbance_update_topic = rospy.get_param("~topics/disturbance_update")
        self.disturbance_update_sub = rospy.Subscriber(
            disturbance_update_topic, HiLoArray, self.callback_disturbance_update
        )

        self.disturbance_lo = np.array(config.disturbance_space["lo"])
        self.disturbance_hi = np.array(config.disturbance_space["hi"])

        self.dynamics = config.dynamics
        self.state_topic = rospy.get_param("~topics/state", "/state_array")
        self.state_sub = rospy.Subscriber(self.state_topic, Array, self.callback_state)
        self.state = None
        self.state_initialized = False

        self.beta_skew = rospy.get_param("~beta_skew", 1.0)  # Defaults to a uniform distribution
        self.seed = 0
        self.random_state = np.random.default_rng(seed=self.seed)
        self.rate = rospy.get_param("~rate", 20)
        self.rospy_rate = rospy.Rate(self.rate)

        # Wait for the state to be initialized
        while not rospy.is_shutdown() and not self.state_initialized:
            self.rospy_rate.sleep()
        self.run()

    def compute_disturbance(self):
        disturbance = (
            self.random_state.beta(self.beta_skew, self.beta_skew, size=self.disturbance_lo.shape)
            * (self.disturbance_hi - self.disturbance_lo)
            + self.disturbance_lo
        )
        per_state_disturbance = self.dynamics.disturbance_matrix(self.state, 0.0) @ disturbance
        return per_state_disturbance

    def run(self):
        while not rospy.is_shutdown():
            per_state_disturbance_msg = Array()
            per_state_disturbance = self.compute_disturbance()
            per_state_disturbance_msg.value = per_state_disturbance.tolist()
            self.pub_disturbance.publish(per_state_disturbance_msg)
            self.rospy_rate.sleep()

    def callback_disturbance_update(self, msg):
        self.disturbance_lo = np.array(msg.lo)
        self.disturbance_hi = np.array(msg.hi)

    def callback_state(self, state_est_msg):
        self.state = np.array(state_est_msg.value)
        self.state_initialized = True


if __name__ == "__main__":
    rospy.init_node("disturbance_node")
    safety_filter = DisturbanceNode()
