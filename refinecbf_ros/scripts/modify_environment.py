#!/usr/bin/env python3

import rospy
from refinecbf_ros.msg import Array, HiLoArray, ValueFunctionMsg
from std_msgs.msg import String
from refinecbf_ros.srv import ModifyEnvironment, ModifyEnvironmentResponse
from refinecbf_ros.config import Config
import numpy as np

class ModifyEnvironmentServer:
    def __init__(self):
        
        # Load configuration
        config = Config(hj_setup=True)

        # TODO Judy: Load the disturbances from yaml through config

        # Set up publishers # TODO: Judy fix actuation_update_pub message type (see below)
        self.actuation_update_pub = rospy.Publisher("/actuation_update", HiLoArray, queue_size=1)
        disturbance_update_topic = rospy.get_param("~topics/disturbance_update")
        self.disturbance_update_pub = rospy.Publisher(disturbance_update_topic, HiLoArray, queue_size=1)

        # Set up service
        modify_environment_service = rospy.get_param("~services/modify_environment")
        rospy.Service(modify_environment_service, ModifyEnvironment, self.handle_modified_environment)

    def update_disturbances(self):
        hi = np.array([4.0])  # TODO Judy to customize
        lo = np.array([-4.0])  # TODO Judy to customize
        # self.disturbance_update_pub.publish(Bool(True))
        self.disturbance_update_pub.publish(HiLoArray(hi=hi, lo=lo))
    
    def handle_modified_environment(self, req):
        '''
        To add disturbances, paste the following in a terminal:
          rosservice call /modify_environment "disturbance"
        '''
        modification_request = req.modification
        # if modification_request == "actuation":
        #     # Get the new actuation values
        #     self.actuation_update_pub.publish()
        if modification_request == "disturbance":
            rospy.loginfo("Start")
            self.update_disturbances()
            rospy.loginfo("Stop")
            output = "Disturbance Activated"
        return ModifyEnvironmentResponse(output)


if __name__ == "__main__":
    rospy.init_node("modify_environment_node")
    modify_environment_server = ModifyEnvironmentServer()
    rospy.spin()
