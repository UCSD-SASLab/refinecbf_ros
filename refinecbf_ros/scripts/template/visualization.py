#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import ColorRGBA
from refinecbf_ros.msg import ValueFunctionMsg, Array
from refinecbf_ros.config import Config
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

class Visualization:

    def __init__(self):
        
        # Config:
        config = Config(hj_setup=True)
        self.grid = config.grid
        
        # Subscriber for SDF:
        obstacle_update_topic = rospy.get_param("~topics/obstacle_update")
        self.obstacle_update_sub = rospy.Subscriber(
            obstacle_update_topic, ValueFunctionMsg, self.callback_sdf
        )

        # Subscriber for vf:
        vf_topic = rospy.get_param("~topics/vf_update")
        self.vf_update_sub = rospy.Subscriber(
            vf_topic, ValueFunctionMsg, self.callback_vf
        )

        # Subscriber for Robot State:
        cbf_state_topic = rospy.get_param("~topics/cbf_state")
        state_sub = rospy.Subscriber(cbf_state_topic, Array, self.callback_state)

        # Publisher for Marker messages
        obstacle_marker_topic = rospy.get_param("~topics/obstacle_marker")
        self.obstacle_marker_publisher = rospy.Publisher(obstacle_marker_topic, Marker, queue_size=10)

        # Publisher for SDF
        sdf_marker_topic = rospy.get_param("~topics/sdf_marker")
        self.sdf_marker_publisher = rospy.Publisher(sdf_marker_topic, Marker, queue_size=10)

        # Publisher for VF
        vf_marker_topic = rospy.get_param("~topics/vf_marker")
        self.vf_marker_publisher = rospy.Publisher(vf_marker_topic, Marker, queue_size=10)

        # load Obstacle and Boundary dictionaries
        self.obstacle_dict = rospy.get_param("~/env/obstacles")
        self.boundary_dict = rospy.get_param("~/env/boundary")

    def obstacle_marker(self,obstacle,obstacle_marker_id):
        raise NotImplementedError("Must Be Sublcassed")
    
    def sdf_marker(self,points,sdf_marker_id):
        raise NotImplementedError("Must Be Sublcassed")
    
    def vf_marker(self,points,vf_marler_id):
        raise NotImplementedError("Must Be Sublcassed")
    
    def zero_level_set_contour(self,vf):
        raise NotImplementedError("Must Be Subclassed")
        
    def add_obstacles(self):

        obstacle_marker_id = 1
        if len(self.obstacle_dict) != 0:
            for obstacle in self.obstacle_dict.values():
                # Create a Marker message for each obstacle
                marker = self.obstacle_marker(obstacle,obstacle_marker_id)
                self.obstacle_marker_publisher.publish(marker)
                obstacle_marker_id = obstacle_marker_id + 1
            
    def update_sdf_contour(self):
        
        sdf_marker_id = 100
        array_points = self.zero_level_set_contour(self.sdf)

        for i in range(len(array_points)):
            marker = self.sdf_marker(array_points[i],sdf_marker_id+i)
            self.sdf_marker_publisher.publish(marker)

    def update_vf_contour(self):
        
        vf_marker_id = 200
        array_points = self.zero_level_set_contour(self.vf)

        for i in range(len(array_points)):
            marker = self.vf_marker(array_points[i],vf_marker_id+i)
            self.vf_marker_publisher.publish(marker)


    def callback_sdf(self,sdf_msg):
        self.sdf = np.array(sdf_msg.vf).reshape(self.grid.shape)

    def callback_vf(self,vf_msg):
        self.vf = np.array(vf_msg.vf).reshape(self.grid.shape)

    def callback_state(self, state_msg):
        self.robot_state = jnp.reshape(np.array(state_msg.value), (-1, 1)).T
        if hasattr(self,'sdf') and hasattr(self,'vf'):
            self.update_vf_contour()
            self.update_sdf_contour()
            self.add_obstacles()
