#!/usr/bin/env python3

import rospy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp
from threading import Lock
from refinecbf_ros.msg import ValueFunctionMsg, HiLoArray
from refinecbf_ros.config import Config
from refinecbf_ros.config import CrazyflieCBF, CrazyflieDynamics #may not need if in launch file param is fed
from refine_cbfs import HJControlAffineDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF, ControlAffineASIF, SlackifiedControlAffineASIF, BatchedDynamics
from refine_cbfs import HJControlAffineDynamics, TabularControlAffineCBF, TabularTVControlAffineCBF, utils
import os

class HJReachabilityNode:
    """
    HJReachabilityNode is a ROS node that computes the Hamilton-Jacobi reachability for a robot.

    Subscribers:
    - disturbance_update_sub (~topics/disturbance_update): Updates the disturbance.
    - actuation_update_sub (~topics/actuation_update): Updates the actuation.
    - obstacle_update_sub (~topics/obstacle_update): Updates the obstacles.

    Publishers:
    - vf_pub (~topics/vf_update): Publishes the value function.
    """

    def __init__(self) -> None:
        """
        Initializes the HJReachabilityNode. It sets up ROS subscribers for disturbance, actuation, and obstacle updates,
        and a publisher for the value function. It also initializes the Hamilton-Jacobi dynamics and the value function.
        """
        # Load configuration
        config = Config(hj_setup=True)

        # Initialize dynamics, grid, and Hamilton-Jacobi dynamics
        self.dynamics = config.dynamics
        self.grid = config.grid
        self.hj_dynamics = config.hj_dynamics

        # Initialize a lock for thread-safe value function updates
        self.vf_lock = Lock()

        # Get initial safe space and setup solver
        self.sdf_values = np.array(
            rospy.wait_for_message(rospy.get_param("~topics/obstacle_update"), ValueFunctionMsg).vf
        ).reshape(self.grid.shape)
        self.brt = lambda sdf_values: lambda t, x: jnp.minimum(x, sdf_values)
        self.solver_settings = hj.SolverSettings.with_accuracy("medium", value_postprocessor=self.brt(self.sdf_values))

        #loading yaml file information
        self.initial_CBF_param= rospy.get_param('~CBF_param', 'initial_CBF_param.yaml')
        
        scaling = rospy.get_param('~scaling', [2.0, 2.0, 1.0, 1.0])
        center = rospy.get_param('~center', [2.0, 2.0, 1.0, 1.0])
        offset = rospy.get_param('~offset', 2.0)


        self.dyn = CrazyflieDynamics(params={'g': 9.81}, dt=0.01, test=False)
        self.batched_dyn = BatchedDynamics(self.dyn)
        
        # self.cbf_cf_params = {"scaling": jnp.array([2.0, 2.0, 1.0, 1.0]), "offset": 2.0, "center": jnp.array([-3.5, 2., 0.0, 0.])}
        self.cbf_cf_params = {"scaling": jnp.array(scaling), "offset": offset, "center": jnp.array(center)}
        self.cbf_cf = CrazyflieCBF(self.batched_dyn, self.cbf_cf_params, test=False)
    
        self.tabular_cbf = TabularControlAffineCBF(self.batched_dyn, self.cbf_cf_params, test=False, grid=self.grid)
        # self.tabular_cbf.tabularize_cbf(self.cbf_cf)
        file_path = os.path.join('/home/sosuke/ros1_ws/crazyflie_clean/ros/src/refinecbf_ros_sk/refinecbf_ros/scripts', 'target_values.npy') #FIXME Hard-coded
        target_values = np.load(file_path)
                                
        self.tabular_cbf = target_values[-1] # Take the last value (when converged)

        # Set up value function publisher
        self.vf_topic = rospy.get_param("~topics/vf_update")
        self.vf_pub = rospy.Publisher(self.vf_topic, ValueFunctionMsg, queue_size=1)

        # Publish initial value function
        self.vf_pub.publish(ValueFunctionMsg(vf=self.tabular_cbf.flatten())) 
        self.update_vf_flag = False #Changed to false for now

        # Set up subscribers for disturbance, actuation, and obstacle updates
        disturbance_update_topic = rospy.get_param("~topics/disturbance_update")
        self.disturbance_update_sub = rospy.Subscriber(
            disturbance_update_topic, HiLoArray, self.callback_disturbance_update
        )

        actuation_update_topic = rospy.get_param("~topics/actuation_update")
        self.actuation_update_sub = rospy.Subscriber(actuation_update_topic, HiLoArray, self.callback_actuation_update)

        obstacle_update_topic = rospy.get_param("~topics/obstacle_update")
        self.obstacle_update_sub = rospy.Subscriber(
            obstacle_update_topic, ValueFunctionMsg, self.callback_obstacle_update
        )

        # Start updating the value function
        self.update_vf()  # This keeps spinning

    def callback_disturbance_update(self, msg):
        """
        Callback for the disturbance update subscriber.

        Args:
            msg (HiLoArray): The incoming disturbance update message.

        This method updates the disturbance space and the dynamics.
        """
        with self.vf_lock:
            max_disturbance = msg.hi
            min_disturbance = msg.lo
            self.disturbance_space = hj.Sets.Box(lo=jnp.array(min_disturbance), hi=jnp.array(max_disturbance))
            self.update_dynamics()  # FIXME:Check whether this is required or happens automatically

    def callback_actuation_update(self, msg):
        """
        Callback for the actuation update subscriber.

        Args:
            msg (HiLoArray): The incoming actuation update message.

        This method updates the control space and the dynamics.
        """
        with self.vf_lock:
            max_control = msg.hi
            min_control = msg.lo
            self.control_space = hj.Sets.Box(lo=jnp.array(min_control), hi=jnp.array(max_control))
            self.update_dynamics()  # FIXME:Check whether this is required or happens automatically

    def callback_obstacle_update(self, msg):
        """
        Callback for the obstacle update subscriber.

        Args:
            msg (ValueFunctionMsg): The incoming obstacle update message.

        This method updates the obstacle and the solver settings.
        """
        with self.vf_lock:
            rospy.loginfo("Updating obstacle")
            self.sdf_values = np.array(msg.vf).reshape(self.grid.shape)
            self.solver_settings = hj.SolverSettings.with_accuracy(
                "medium", value_postprocessor=self.brt(self.sdf_values)
            )

    def update_dynamics(self):
        """
        Updates the Hamilton-Jacobi dynamics based on the current control and disturbance spaces.
        """
        self.hj_dynamics = HJControlAffineDynamics(
            self.dynamics, control_space=self.control_space, disturbance_space=self.disturbance_space
        )

    def update_vf(self):
        """
        Continuously updates the value function and publishes it as long as the node is running and the update flag is set.
        """
        while self.update_vf_flag and not rospy.is_shutdown():
            with self.vf_lock:
                new_values = hj.step(
                    self.solver_settings, self.hj_dynamics, self.grid, 0.0, self.vf.copy(), -0.1, progress_bar=False
                )
                self.vf = new_values
                vf_msg = ValueFunctionMsg()
                vf_msg.vf = np.array(self.vf).flatten()
                self.vf_pub.publish(vf_msg)  # FIXME: Nate figure out better way
            rospy.sleep(1)  # To make sure that subscribers can run

if __name__ == "__main__":
    rospy.init_node("hj_reachability_node")
    HJReachabilityNode()
