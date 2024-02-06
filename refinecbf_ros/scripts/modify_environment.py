#!/usr/bin/env python3

import rospy
from refinecbf_ros.msg import Array, HiLoArray, ValueFunctionMsg
from refinecbf_ros.srv import ModifyEnvironment, ModifyEnvironmentResponse
from refinecbf_ros.config import Config

class ModifyEnvironmentServer:
    def __init__(self):
        
        # Load configuration
        config = Config(hj_setup=True)

        # Initialize dynamics, grid, and Hamilton-Jacobi dynamics
        self.disturbance_space = config.disturbance_space

        # Set up publishers
        self.actuation_update_pub = rospy.Publisher("/actuation_update", HiLoArray, queue_size=1)
        self.disturbance_update_pub = rospy.Publisher("/disturbance_update", HiLoArray, queue_size=1)

        # Set up service
        modify_environment_service = rospy.get_param("~services/modify_environment")
        rospy.Service(modify_environment_service, ModifyEnvironment, self.handle_modified_environment)

    def handle_modified_environment(self, req):
        modification_request = req.modification
        # if modification_request == "actuation":
        #     # Get the new actuation values
        #     self.actuation_update_pub.publish()
        if modification_request == "disturbance":
            self.disturbance_update_pub.publish(self.disturbance_space)
            output = "Disturbance Activated"
        return ModifyEnvironmentResponse(output)


if __name__ == "__main__":
    rospy.init_node("modify_environment_node")
    modify_environment_server = ModifyEnvironmentServer()
