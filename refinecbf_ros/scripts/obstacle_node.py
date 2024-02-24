#!/usr/bin/env python3

import rospy
import numpy as np
import jax.numpy as jnp
import jax
import hj_reachability as hj
from refinecbf_ros.msg import Array, ValueFunctionMsg, Obstacles
from refinecbf_ros.config import Config
from refinecbf_ros.srv import ActivateObstacle, ActivateObstacleResponse
from std_msgs.msg import Bool
import pdb
import matplotlib.pyplot as plt


class ObstacleNode:
    def __init__(self) -> None:
        # Following publishers:
        # - /env/obstacle_update
        # Following subscribers:
        # - /state

        # Config:
        config = Config(hj_setup=True)
        self.dynamics = config.dynamics
        self.grid = config.grid
        self.detection_obstacles = config.detection_obstacles
        self.service_obstacles = config.service_obstacles
        self.active_obstacles = config.active_obstacles
        self.update_obstacles = config.update_obstacles
        self.active_obstacle_names = config.active_obstacle_names
        self.boundary = config.boundary

        # Publishers:
        self.vf_update_method = rospy.get_param("~vf_update_method")

        sdf_update_topic = rospy.get_param("~topics/sdf_update", "/env/sdf_update")
        if self.vf_update_method == "pubsub":
            self.sdf_update_pub = rospy.Publisher(sdf_update_topic, ValueFunctionMsg, queue_size=1)
        elif self.vf_update_method == "file":
            self.sdf_update_pub = rospy.Publisher(sdf_update_topic, Bool, queue_size=1)
        else:
            raise NotImplementedError("{} is not a valid vf update method".format(self.vf_update_method))
        
        obstacle_update_topic = rospy.get_param("~topics/obstacle_update")
        self.obstacle_update_pub = rospy.Publisher(obstacle_update_topic,Obstacles,queue_size=1)

        # Subscribers:
        cbf_state_topic = rospy.get_param("~topics/cbf_state")
        state_sub = rospy.Subscriber(cbf_state_topic, Array, self.callback_state)

        # Wait for state message before proceeding
        rospy.wait_for_message(cbf_state_topic,Array)

        # Services:
        activate_obstacle_service = rospy.get_param("~services/activate_obstacle")
        rospy.Service(activate_obstacle_service, ActivateObstacle, self.handle_activate_obstacle)

        # Initialize SDF(just active obstacles + boundary):
        self.update_sdf()  # Publish initial sdf

        # Set start time
        self.startTime = rospy.Time().now().to_sec()

    def obstacle_detection(self):
        updatesdf = False
        for obstacle in self.detection_obstacles:
            if obstacle not in self.active_obstacles:
                if obstacle.distance_to_obstacle(self.robot_state) <= obstacle.detectionRadius:
                    self.active_obstacles.append(obstacle)
                    self.active_obstacle_names.append(obstacle.obstacleName)
                    updatesdf = True
        for obstacle in self.update_obstacles:
            if obstacle not in self.active_obstacles:
                if obstacle.updateTime >= rospy.Time.now().to_sec() - self.startTime:
                    self.active_obstacles.append(obstacle)
                    self.active_obstacle_names.append(obstacle.obstacleName)
                    updatesdf = True

        if updatesdf:
            self.update_sdf()
            self.update_active_obstacles()

    def update_sdf(self):
        sdf = hj.utils.multivmap(self.build_sdf(), jnp.arange(self.grid.ndim))(self.grid.states)
        rospy.loginfo("Here!")
        if self.vf_update_method == "pubsub":
            self.sdf_update_pub.publish(ValueFunctionMsg(sdf.flatten()))
        else:  # self.vf_update_method == "file"
            np.save("./sdf.npy", sdf)
            self.sdf_update_pub.publish(Bool(True))

    def update_active_obstacles(self):
        self.obstacle_update_pub.publish(Obstacles(self.active_obstacle_names))

    def callback_state(self, state_msg):
        self.robot_state = jnp.reshape(np.array(state_msg.value), (-1, 1))

    def build_sdf(self):
        def sdf(x):
            sdf = self.boundary.boundary_sdf(x)
            for obstacle in self.active_obstacles:
                obstacle_sdf = obstacle.obstacle_sdf(x)
                sdf = jnp.min(jnp.array([sdf, obstacle_sdf]))
            return sdf

        return sdf

    def handle_activate_obstacle(self, req):
        obstacle_index = req.obstacleNumber
        if obstacle_index >= len(self.service_obstacles):
            output = "Invalid Obstacle Number"
        elif self.service_obstacles[obstacle_index] in self.active_obstacles:
            output = "Obstacle Already Active"
        else:
            self.active_obstacles.append(self.service_obstacles[obstacle_index])
            self.update_sdf()
            output = "Obstacle Activated"
        return ActivateObstacleResponse(output)


if __name__ == "__main__":
    rospy.init_node("obstacle_node")
    ObstacleNodeObject = ObstacleNode()

    rate = rospy.Rate(rospy.get_param("~/env/obstacle_update_rate_hz"))

    while not rospy.is_shutdown():
        ObstacleNodeObject.obstacle_detection()
        rate.sleep()
