#!/usr/bin/env python3

import rospy
import numpy as np
import jax.numpy as jnp
from refinecbf_ros.msg import ValueFunctionMsg, Array, HiLoArray
from cbf_opt import ControlAffineASIF
from refine_cbfs import TabularControlAffineCBF
from refinecbf_ros.config import Config
from refinecbf_ros.config import CrazyflieCBF, CrazyflieDynamics
from cbf_opt import ControlAffineDynamics, ControlAffineCBF, ControlAffineASIF, SlackifiedControlAffineASIF, BatchedDynamics

class SafetyFilterNode:
    """
    Balances safety filter and
    """
    def __init__(self):
        self.safety_filter_active = rospy.get_param("~safety_filter_active", True)
        vf_topic = rospy.get_param("~topics/vf_update", "/vf_update")
        self.vf_sub = rospy.Subscriber(vf_topic, ValueFunctionMsg, self.callback_vf_update)
        self.state_topic = rospy.get_param("~topics/state", "/state_array")
        self.state_sub = rospy.Subscriber(self.state_topic, Array, self.callback_state)
        
        config = Config(hj_setup=True)
        self.dynamics = config.dynamics #FIXME: Likely cause of problem
        self.grid = config.grid

        self.dyn = CrazyflieDynamics(params={'g': 9.81}, dt=0.01, test=False)
        self.batched_dyn = BatchedDynamics(self.dyn)

        self.cbf_cf_params = {"scaling": jnp.array([2.0, 2.0, 1.0, 1.0]), "offset": 2.0, "center": jnp.array([-3.5, 2., 0.0, 0.])}
        self.cbf_cf = CrazyflieCBF(self.batched_dyn, self.cbf_cf_params, test=False)

        self.cbf = TabularControlAffineCBF(self.dynamics, grid=self.grid)
        self.safety_filter_solver = ControlAffineASIF(self.dynamics, self.cbf)
       
        self.cbf.tabularize_cbf(self.cbf_cf) #FIXME Leading to assert error

        self.safety_filter_solver.umin = np.array(config.control_space["lo"])
        self.safety_filter_solver.umax = np.array(config.control_space["hi"])

        nom_control_topic = rospy.get_param("~topics/nominal_control", "/control/nominal")
        self.nominal_control_sub = rospy.Subscriber(nom_control_topic, Array, self.callback_safety_filter)
        self.state = None
        filtered_control_topic = rospy.get_param("~topics/filtered_control", "/control/filtered")
        self.pub_filtered_control = rospy.Publisher(filtered_control_topic, Array, queue_size=1)

        actuation_update_topic = rospy.get_param("~topics/actuation_update", "/env/actuation_update")
        self.actuation_update_sub = rospy.Subscriber(actuation_update_topic, HiLoArray, 
                                                     self.callback_actuation_update)

        if self.safety_filter_active:
            # This has to be done to ensure real-time performance
            self.initialized_safety_filter = False
            self.safety_filter_solver.setup_optimization_problem()
            rospy.logwarn("safety filter is active!")

        else:
            self.initialized_safety_filter = True
            self.safety_filter_solver = lambda state, nominal_control: nominal_control
            rospy.logwarn("No safety filter, be careful!")


    def callback_actuation_update(self, msg):
        self.safety_filter_solver.umin = np.array(msg.lo)
        self.safety_filter_solver.umax = np.array(msg.hi)

    def callback_vf_update(self, vf_msg):
        self.cbf.vf_table = np.array(vf_msg.vf).reshape(self.grid.shape)
        print("Updated vf")
        if not self.initialized_safety_filter:
            rospy.loginfo("Initialized safety filter")
            self.initialized_safety_filter = True

    def callback_safety_filter(self, control_msg):
        nom_control = np.array([control_msg.value])
        if self.state is None:
            rospy.loginfo(" State not set yet, no control published")
            return
        if not self.initialized_safety_filter: #if initialzied_safety_filter=False, goes here which we don't want
            safety_control_msg = control_msg
            rospy.loginfo("Safety filter not initialized, gonna use nominal")
        else:
            safety_control_msg = Array()
            vf = np.array(self.safety_filter_solver.cbf.vf(self.state.copy(), 0.0)).item() #used to be commented and placed below safet_control =self.
            rospy.loginfo("value at current state:{}".format(vf)) #Used to be commented
            safety_control = self.safety_filter_solver(self.state.copy(), nominal_control=nom_control)
            safety_control_msg.value = safety_control[0].tolist()  # Ensures compatibility

        self.pub_filtered_control.publish(safety_control_msg)

    def callback_state(self, state_est_msg):
        self.state = np.array(state_est_msg.value)


if __name__ == "__main__":
    rospy.init_node("safety_filter_node")
    safety_filter = SafetyFilterNode()
    rospy.spin()