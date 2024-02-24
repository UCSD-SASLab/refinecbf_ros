#!/usr/bin/env python3

import rospy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp
from std_msgs.msg import Empty

from refinecbf_ros.config import Config

# add the parent of this folder tot he path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from template.nominal_controller import NominalController


class CrazyflieNominalControl(NominalController):
    def __init__(self):
        super().__init__()
        self.config = Config(hj_setup=False)
        self.dynamics = self.config.dynamics
        # Control bounds
        self.max_thrust = rospy.get_param("~control/max_thrust")
        self.min_thrust = rospy.get_param("~control/min_thrust")
        self.max_roll = rospy.get_param("~control/max_roll")
        self.max_pitch = rospy.get_param("~control/max_pitch")
        self.safety_controls_idis = self.config.safety_controls

        self.target = jnp.array(rospy.get_param("/ctr/nominal/goal/coordinates"))
        self.state = np.zeros_like(self.target)
        # assert len(self.target) == self.dynamics.n_dims  # TODO: Different check needed

        utarget_file = rospy.get_param("~LQR/u_ref_file")
        self.u_target = np.loadtxt(utarget_file)
        # assert len(self.u_target) == self.dynamics.control_dims  # TODO: Different check needed

        # Initialize parameters
        self.gain = np.array(rospy.get_param("/ctr/nominal/goal/gain"))
        # assert self.gain.shape == (self.dynamics.control_dims, self.dynamics.n_dims)  # TODO: Diff check needed
        umin = np.array([-self.max_roll, -self.max_pitch, -np.inf, self.min_thrust])
        umax = np.array([self.max_roll, self.max_pitch, np.inf, self.max_thrust])
        umin[self.safety_controls_idis] = np.array(self.config.control_space["lo"])
        umax[self.safety_controls_idis] = np.array(self.config.control_space["hi"])
        self.controller = lambda x, t: np.clip(self.u_target + self.gain @ (x - self.target), umin, umax)


if __name__ == "__main__":
    rospy.init_node("cf_nominal_control_node")
    Controller = CrazyflieNominalControl()

    rate = rospy.Rate(rospy.get_param("/ctr/nominal/frequency"))

    while not rospy.is_shutdown():
        Controller.publish_control()
        rate.sleep()
