#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import ColorRGBA
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # FIXME: Make it so that the directory is autoatically in search path
import numpy as np
from template.visualization import Visualization
import matplotlib
matplotlib.use('Agg')


class CrazyflieVisualization(Visualization):

    def __init__(self):
        
        super().__init__()

    def obstacle_marker(self,obstacle,obstacle_marker_id,active):
        marker = Marker()
        marker.header.frame_id = "world"
        if obstacle["type"] == "Circle":
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            # Set obstacle parameters
            marker.scale.x = 2.0*obstacle['radius']
            marker.scale.y = 2.0*obstacle['radius']
            marker.scale.z = .5

            marker.color = ColorRGBA()
            if active:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.75
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.5

            marker.pose.position.y = obstacle['center'][0]
            marker.pose.position.z = obstacle['center'][1]
            marker.pose.position.x = 0.0

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = (2.0 ** .5)/2.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = (2.0 ** .5)/2.0

            marker.id = obstacle_marker_id

        elif obstacle["type"] == "Rectangle":
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set obstacle parameters
            marker.scale.x = obstacle['maxVal'][1]-obstacle['minVal'][1]
            marker.scale.y = obstacle['maxVal'][0]-obstacle['minVal'][0]
            marker.scale.z = 0.5

            marker.color = ColorRGBA()
            if active:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.75
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.5

            marker.pose.position.y = (obstacle['maxVal'][0]+obstacle['minVal'][0])/2.0
            marker.pose.position.z = (obstacle['maxVal'][1]+obstacle['minVal'][1])/2.0
            marker.pose.position.x = 0.0

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = (2.0 ** .5)/2.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = (2.0 ** .5)/2.0

            marker.id = obstacle_marker_id

        else:
            raise ValueError("Invalid Obstacle Type: {}".format(obstacle["type"]))
        
        return marker
    
    def sdf_marker(self,points,sdf_marker_id):
        marker = Marker()
        marker.header.frame_id="world"

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = .1 # Line Width

        marker.color = ColorRGBA()
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        marker.id = sdf_marker_id

        marker.points = [Point(0.0,x,y) for x,y in points]

        return marker
    
    def vf_marker(self,points,vf_marker_id):
        marker = Marker()
        marker.header.frame_id="world"

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = .1 # Line Width

        marker.color = ColorRGBA()
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.id = vf_marker_id

        marker.points = [Point(0.0,x,y) for x,y in points]

        return marker
    
    def goal_marker(self, control_dict,goal_marker_id):
        marker = Marker()
        marker.header.frame_id = "world"

        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = .1 # Point Width
        marker.scale.y = .1

        marker.color = ColorRGBA()
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        point = Point()
        point.x = 0.0
        point.y = np.array(control_dict['goal']['coordinates'])[self.state_safety_idis][0]
        point.z = np.array(control_dict['goal']['coordinates'])[self.state_safety_idis][1]
        marker.points.append(point)

        marker.id = goal_marker_id

        return marker

    
    def zero_level_set_contour(self,vf):
        robot_state = self.clip_state(self.robot_state)
        contour = plt.contour(self.grid.coordinate_vectors[0], 
                              self.grid.coordinate_vectors[1], 
                              vf[:, :, self.grid.nearest_index(robot_state)[0][2],self.grid.nearest_index(robot_state)[0][3]].T, levels=[0])
        array_points = [path.vertices for path in contour.collections[0].get_paths()]
        return array_points


if __name__ == '__main__':
    rospy.init_node('tb_visualization', anonymous=True)
    CrazyflieVisualization()
    rospy.spin()
