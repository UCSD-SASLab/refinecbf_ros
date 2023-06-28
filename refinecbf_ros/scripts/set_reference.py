#!/usr/bin/env python3

import rospy
from crazyflie_msgs.msg import PositionVelocityStateStamped, PositionVelocityYawStateStamped, PrioritizedControlStamped
from refinecbf_ros.msg import Control2D, State2D, Control2DTimeLimited 


class Process2DData:
    def __init__(self):
        control_topic = rospy.get_param("~topics/prioritized_control", "/control/raw")
        ref_state_topic = rospy.get_param("~topics/reference", "/ref")
        self.pub_control = rospy.Publisher(control_topic, PrioritizedControlStamped, queue_size=1)
        self.pub_ref_state = rospy.Publisher(ref_state_topic, PositionVelocityStateStamped, queue_size=10)
        self.sub_control = rospy.Subscriber("/control_2d", Control2DTimeLimited, self.callback_new_control)
        self.sub_state = rospy.Subscriber("/state_2d", State2D, self.callback_state2d)
        
        self.controller_rate = rospy.get_param("~estimator_dt", 0.01)
        self.timer = rospy.Timer(rospy.Duration(self.controller_rate), self.callback_control_timer)

        self.current_control = PrioritizedControlStamped()
        self.control_duration = 0  # int
        self.control_start_time = None

    def callback_new_control(self, req):
        self.current_control = PrioritizedControlStamped()
        self.current_control.header.stamp = rospy.Time.now()
        self.current_control.control.priority = 1.0
        self.current_control.control.control.roll = req.roll 
        self.current_control.control.control.pitch = 0.0
        self.current_control.control.control.yaw_dot = 0.0
        self.current_control.control.control.thrust = req.thrust
        self.control_duration = req.duration
        self.control_start_time = rospy.Time.now()

    def callback_control_timer(self, timer):
        if self.control_start_time is None or rospy.Time.now() - self.control_start_time > rospy.Duration(self.control_duration):
            self.current_control.control.priority = 0.0  # Reset the priority back to zero
        self.pub_control.publish(self.current_control)

    def callback_state2d(self, req):
        reference = PositionVelocityStateStamped()
        reference.header.stamp = rospy.Time.now()
        reference.state.x = 0.0
        reference.state.y = req.y
        reference.state.z = req.z
        reference.state.x_dot = 0.0
        reference.state.y_dot = req.y_dot
        reference.state.z_dot = req.z_dot
        self.pub_ref_state.publish(reference)

if __name__ == "__main__":
    rospy.init_node("interface")
    Process2DData()
    rospy.spin()

