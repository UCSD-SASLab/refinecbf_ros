#!/usr/bin/env python3

import rospy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp

from refinecbf_ros.msg import StateArray, ControlArray
from nominal_hjr_control import NominalControlHJNP
from refinecbf_ros.config import Config

class TurtlebotNominalControl:
    """
    This class determines the nominal control of the turtlebot using hj_reachability in refineCBF
    """

    def __init__(self):
        # Subscribers:
        #   StateArray - Turtlebot State
        # Publishers:
        #   ControlArray - Turtlebot Control

        # Topics
        cbf_state_topic = rospy.get_param("~topics/state")
        robot_nominal_control_topic = rospy.get_param("~topics/nominal_control")

        self.state_sub = rospy.Subscriber(cbf_state_topic, StateArray, self.callback_state)
        self.control_pub = rospy.Publisher(robot_nominal_control_topic, ControlArray, queue_size=1)

        # Dynamics
        config = Config(hj_setup=True)
        dynamics = config.dynamics 
        hj_dynamics = config.hj_dynamics

        # Controller Domain, Target, Grid and Padding
        state_domain = hj.sets.Box(lo=jnp.array(rospy.get_param("/ctr/grid_domain/lo")),hi=jnp.array(rospy.get_param("/ctr/grid_domain/hi")))
        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, jnp.array(rospy.get_param("/ctr/grid_domain/resolution")), periodic_dims=2)
        target = jnp.array(rospy.get_param("/ctr/goal/coordinates"))
        padding = jnp.array(rospy.get_param("/ctr/goal/padding"))
        time_horizon = jnp.array(rospy.get_param("/ctr/time/time_horizon"))
        self.time_index = jnp.array(rospy.get_param("/ctr/time/time_index"))

        # Initialize Controller
        self.Controller = NominalControlHJNP(hj_dynamics,grid,final_time = time_horizon, time_intervals = 101,solver_accuracy="low",target=target,padding = padding)
        self.Controller.solve()

    def callback_state(self,state_array_msg):
        self.state = np.array(state_array_msg.state)

    def publish_control(self):
        control = self.Controller.get_nominal_control(self.state,self.time_index)
        control_msg = ControlArray()
        control_msg.control = [control[0][0],control[0][1]]
        self.control_pub.publish(control_msg)


if __name__ == "__main__":
    rospy.init_node("tb3_nominal_control_node")
    Controller = TurtlebotNominalControl()

    rate = rospy.Rate(rospy.get_param("/ctr/time/rate"))

    while not rospy.is_shutdown():
        Controller.publish_control()
        rate.sleep()

