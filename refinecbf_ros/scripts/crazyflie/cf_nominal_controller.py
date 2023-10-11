#!/usr/bin/env python3

import rospy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp

from refinecbf_ros.config import Config

# add the parent of this folder tot he path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from template.nominal_controller import NominalController


class CrazyflieNominalControl(NominalController):
    def __init__(self):
        super().__init__()
        self.config =  Config(hj_setup=False)
        self.dynamics = self.config.dynamics
        
        self.state = np.zeros(self.dynamics.n_dims)
        
        self.target = jnp.array(rospy.get_param("/ctr/nominal/goal/coordinates"))
        assert len(self.target) == self.dynamics.n_dims

        self.u_target = jnp.array(rospy.get_param("/ctr/nominal/goal/control"))
        assert len(self.u_target) == self.dynamics.control_dims

        self.gain = jnp.array(rospy.get_param("/ctr/nominal/goal/gain"))
        assert self.gain.shape == (self.dynamics.control_dims, self.dynamics.n_dims)

        self.controller = lambda x, t: self.u_target + self.gain @ (x - self.target)


if __name__ == "__main__":
    rospy.init_node("tb3_nominal_control_node")
    Controller = CrazyflieNominalControl()

    rate = rospy.Rate(rospy.get_param("/ctr/nominal/frequency"))

    while not rospy.is_shutdown():
        Controller.publish_control()
        rate.sleep()
