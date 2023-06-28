#!/usr/bin/env python3

import rospy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp
from threading import Lock
from cfs_interface.msg import ControlArray, StateArray, ValueFunctionMsg, HiLoStateArray
from cfs_interface.config import Config


from refine_cbfs import HJControlAffineDynamics


class HJReachabilityNode:
    def __init__(self) -> None:
        # Following publishers:
        # - /vf
        # Following subscribers:
        # - /env/disturbance_update
        # - /env/actuation_update
        # - /env/obstacle_update    
        config = Config(hj_setup=True)
        self.dynamics = config.dynamics
        self.grid = config.grid

        self.hj_dynamics = config.hj_dynamics 

        self.vf_lock = Lock()

        self.sdf = self.setup_safe_space(config=config)
        self.sdf_values = hj.utils.multivmap(self.sdf, jnp.arange(self.grid.ndim))(self.grid.states)
        initial_vf_file = rospy.get_param("~init_vf_file", None)
        if initial_vf_file is None:
            self.vf = self.sdf_values
        else:
            self.vf = np.load(initial_vf_file)

        self.brt = lambda sdf_values: lambda t, x: jnp.minimum(x, sdf_values)
        # should this be updated when the obstacle is updated or will this happen automatically?
        self.solver_settings = hj.SolverSettings.with_accuracy("medium", value_postprocessor=self.brt(self.sdf_values))

        self.vf_topic = rospy.get_param("~topics/vf_update", "/vf_update")
        self.vf_pub = rospy.Publisher(self.vf_topic, ValueFunctionMsg, queue_size=1)
        
        rospy.sleep(5)
        self.vf_pub.publish(ValueFunctionMsg(vf=self.vf.flatten()))
        self.update_vf_flag = True

        # Setting up the subscribers
        disturbance_update_topic = rospy.get_param("~topics/disturbance_update", "/env/disturbance_update")
        self.disturbance_update_sub = rospy.Subscriber(disturbance_update_topic, HiLoStateArray, 
                                                       self.callback_disturbance_update)

        actuation_update_topic = rospy.get_param("~topics/actuation_update", "/env/actuation_update")
        self.actuation_update_sub = rospy.Subscriber(actuation_update_topic, HiLoStateArray, 
                                                     self.callback_actuation_update)

        obstacle_update_topic = rospy.get_param("~topics/obstacle_update", "/env/obstacle_update")
        self.obstacle_update_sub = rospy.Subscriber(obstacle_update_topic, ValueFunctionMsg, 
                                                    self.callback_obstacle_update)


        # Rospy make sure that subscribers run in separate thread from update_vf
        self.update_vf()  # This keeps spinning
            
    def setup_safe_space(self, config):  # TODO: This has to be recalled when an obstacle moves or disappears or appears
        bouding_box = self.setup_bounding_box(config)
        combined_array = lambda state: bouding_box(state)
        for obstacle in config.obstacles:
            obstacle_array = self.setup_obstacle(obstacle)
            combined_array = lambda state: jnp.min(jnp.array([combined_array, obstacle_array(state)]))
        return combined_array


    def setup_bounding_box(self, config):
        return lambda state: jnp.min(jnp.concatenate([state - jnp.array(config.safe_set["lo"]), 
                                                      jnp.array(config.safe_set["hi"]) - state]))
    
    def setup_obstacle(self, obstacle):
        return lambda state: -jnp.min(jnp.concatenate([state - jnp.array(obstacle["lo"]), 
                                                       jnp.array(obstacle["hi"]) - state]))


    def callback_disturbance_update(self, msg):
        with self.vf_lock:
            max_disturbance = msg.hi
            min_disturbance = msg.lo
            self.disturbance_space = hj.Sets.Box(lo=jnp.array(min_disturbance), hi=jnp.array(max_disturbance))
            self.update_dynamics()

    def callback_actuation_update(self, msg):
        with self.vf_lock:
            max_control = msg.hi
            min_control = msg.lo
            self.control_space = hj.Sets.Box(lo=jnp.array(min_control), hi=jnp.array(max_control))
            self.update_dynamics()  # Check whether this is required or happens automatically

    def callback_obstacle_update(self, msg):
        # msg is a value function
        with self.vf_lock:
            rospy.loginfo("Updating obstacle")
            self.obstacle = np.array(msg.vf).reshape(self.grid.shape)
            self.solver_settings = hj.SolverSettings.with_accuracy("medium", value_postprocessor=self.brt(self.obstacle))

    def update_dynamics(self):
        self.hj_dynamics = HJControlAffineDynamics(self.dynamics, control_space=self.control_space, disturbance_space=self.disturbance_space)

    def update_vf(self):
        while self.update_vf_flag and not rospy.is_shutdown():
            with self.vf_lock:
                new_values = hj.step(self.solver_settings, self.hj_dynamics, self.grid, 0.0, self.vf.copy(), -0.1, progress_bar=False)
                self.vf = new_values
                vf_msg = ValueFunctionMsg()
                vf_msg.vf = self.vf.flatten()
                self.vf_pub.publish(vf_msg)  # TODO: Fix, this operation is really slow, but camera images are not necessarily that much worse... (although ints)
            rospy.sleep(0.001)  # To make sure that subscribers can run


if __name__ == "__main__":
    rospy.init_node("hj_reachability_node")
    HJReachabilityNode()