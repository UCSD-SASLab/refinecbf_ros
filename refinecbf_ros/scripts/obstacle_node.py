import rospy
import numpy as np
import jax.numpy as jnp
import jax
from refinecbf_ros.msg import StateArray,ValueFunctionMsg
from refinecbf_ros.config import Config
from refinecbf_ros.srv import ActivateObstacle, ActivateObstacleResponse

class ObstacleDetectionNode:

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
        self.update_obstacles = config.update_obstacles
        self.boundary = config.boundary
        
        # Publishers:
        obstacle_update_topic = rospy.get_param("~topics/obstacle_update_topic", "/env/obstacle_update")
        self.obstacle_update_pub = rospy.Publisher(obstacle_update_topic, ValueFunctionMsg, queue_size=1)

        # Subscribers:
        robot_state_topic = rospy.get_param("~topics/robot_state", "/state")
        state_sub = rospy.Subscriber(robot_state_topic, StateArray, self.callback_state)

        # Services:
        activate_obstacle_service = rospy.get_param("~services/activate_obstacle")
        rospy.Service(activate_obstacle_service,ActivateObstacle,self.handle_activate_obstacle)

        # Initialize Active Obstacles (Just Boundary):
        self.active_obstacles = []
        sdf_msg = ValueFunctionMsg()
        sdf_msg.vf = self.build_sdf(self.active_obstacles,self.boundary)
        self.obstacle_update_pub.publish(sdf_msg)


    def obstacle_detection(self):
        updatesdf = False
        for obstacle in self.detection_obstacles:
            if obstacle not in self.active_obstacles:
                if obstacle.distance_to_obstacle(self.robot_state) <= obstacle.detectionRadius:
                    self.active_obstacles.append(obstacle)
                    updatesdf = True
        for obstacle in self.update_obstacles: 
            if obstacle not in self.active_obstacles:
                if obstacle.updateTime >= rospy.Time.now():
                    self.active_obstacles.append(obstacle)
                    updatesdf = True
        
        if updatesdf:
            self.update_sdf()

    def update_sdf(self):
        sdf_msg = ValueFunctionMsg()
        sdf_msg.vf = self.build_sdf(self.active_obstacles,self.boundary)
        self.obstacle_update_pub.publish(sdf_msg)    

    def callback_state(self, state_msg):
        self.robot_state = np.array(state_msg.state)

    def build_sdf(boundary, obstacles):
        """
        Args:
            boundary: [n x 2] matrix indicating upper and lower boundaries of safe space
            obstacles: list of [n x 2] matrices indicating obstacles in the state space
        Returns:
            Function that can be queried for unbatched state vector
        """
        def sdf(x):
            sdf = boundary.boundary_sdf(x)
            for obstacle in obstacles:
                obstacle_sdf = obstacle.obstacle_sdf(x)
                sdf = jnp.min(jnp.array([sdf, obstacle_sdf]))
            return sdf
        return sdf

    def handle_activate_obstacle(self,req):
        obstacle_index = req.obstacleNumber
        if obstacle_index >= len(self.service_obstacles):
            output = "Invalid Obstacle Number"
        elif self.service_obstacles[obstacle_index] in self.active_obstacles:
            output = "Obstacle Already Active"
        else:
            self.active_obstacles.append(self.service_obstacles[obstacle_index])
            self.update_sdf()
            output = "Obstacle Activated"


if __name__ == "__main__":
    rospy.init_node("obstacle_detection_node")
    ObstacleDetection = ObstacleDetectionNode()

    rate = rospy.Rate(rospy.get_param("~/env/obstacle_update_rate_hz"))

    while not rospy.is_shutdown():
        ObstacleDetection.obstacle_detection()
        rate.sleep()