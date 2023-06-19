#!/usr/bin/env python3

import rospy
from crazyflie_msgs.msg import PositionVelocityStateStamped, PositionVelocityYawStateStamped, ControlStamped
from cfs_interface.msg import Control2D, State2D 


class Process2DData:
    def __init__(self):
        control_topic = rospy.get_param("topics/control", "/control/lqr")
        ref_state_topic = rospy.get_param("topics/reference", "/ref")
        self.pub_ref_control = rospy.Publisher(control_topic, ControlStamped, queue_size=1)
        self.pub_ref_state = rospy.Publisher(ref_state_topic, PositionVelocityStateStamped, queue_size=10)
        self.sub_control = rospy.Subscriber("/control_2d", Control2D, self.callback_control2d)
        self.sub_state = rospy.Subscriber("/state_2d", State2D, self.callback_state2d)

    def callback_control2d(self, req):
        reference = ControlStamped()
        reference.header.stamp = rospy.Time.now()
        reference.control.roll = 0.0 
        reference.control.pitch = req.pitch
        reference.control.yaw_dot = 0.0
        reference.control.thrust = req.thrust
        self.pub_ref_control.publish(reference)

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
    sub1 = rospy.Subscriber("")
