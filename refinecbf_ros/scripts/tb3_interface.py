#!/usr/bin/env python3

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from refinecbf_ros.msg import StateArray, ControlArray

class TurtlebotInterface:
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the Turtlebots. 
    Each HW platform should have its own Interface node
    """

    def __init__(self):
        # Subscribers:
        #   StateOdom - Turtlebot State
        #   X NominalControlIn - Turtlebot Nominal Control
        #   SafeControlIn - Filtered Control from refinecbf
        # Publishers:
        #   StateArray - Turtlebot State
        #   X NominalControlOut - Turtlebot Nominal Control for refinecbf
        #   SafeControlOut - Filtered Control for turtlebot

        # State Array Subscriber
        state_odom_topic = rospy.get_param("~topics/robot_state")
        state_odom_sub = rospy.Subscriber(state_odom_topic,Odometry,self.callback_state_odom)

        # Safe Control In Subscriber
        safe_control_in_topic = rospy.get_param("~topics/cbf_safe_control")
        safe_control_in_sub = rospy.Subscriber(safe_control_in_topic,ControlArray,self.callback_safe_control_in)

        # State Array Publisher
        state_array_topic = rospy.get_param("~topics/cbf_state")
        self.state_array_pub = rospy.Publisher(state_array_topic,StateArray,queue_size=1)

        # Safe Control Out Publisher
        safe_control_out_topic = rospy.get_param("~topics/robot_safe_control")
        self.safe_control_out_pub = rospy.Publisher(safe_control_out_topic, Twist, queue_size=1)

    def callback_state_odom(self,odom_msg):

        w = odom_msg.pose.pose.orientation.w
        x = odom_msg.pose.pose.orientation.x
        y = odom_msg.pose.pose.orientation.y
        z = odom_msg.pose.pose.orientation.z

        # Convert Quaternion to Yaw
        yaw = np.arctan2(2*(w*z+x*y),1-2*(np.power(y,2)+np.power(z,2)))

        state_array_msg = StateArray()
        state_array_msg.state = [odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y,yaw]
        self.state_array_pub.publish(state_array_msg)

    def callback_safe_control_in(self,safe_control_in_msg):
        safe_control_out_msg = Twist()

        safe_control_out_msg.linear.x = safe_control_in_msg.control[1]
        safe_control_out_msg.linear.y = 0.0
        safe_control_out_msg.linear.z = 0.0

        safe_control_out_msg.angular.x = 0.0
        safe_control_out_msg.angular.y = 0.0
        safe_control_out_msg.angular.z = safe_control_in_msg.control[0]
        self.safe_control_out_pub.publish(safe_control_out_msg)



if __name__ == "__main__":
    rospy.init_node("interface")
    TurtlebotInterface()
    rospy.spin()