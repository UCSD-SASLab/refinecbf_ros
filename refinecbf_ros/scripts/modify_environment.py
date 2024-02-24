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
        self.disturbance_space = config.disturbance_space
        self.control_space = config.control_space
        self.actuation_update_list = config.actuation_updates_list
        self.actuation_idx = 0
        self.disturbance_update_list = config.disturbance_updates_list
        self.disturbance_idx = 0

        # Set up publishers
        actuation_update_topic = rospy.get_param("~topics/actuation_update")
        self.actuation_update_pub = rospy.Publisher(actuation_update_topic, HiLoArray, queue_size=1)

        disturbance_update_topic = rospy.get_param("~topics/disturbance_update")
        self.disturbance_update_pub = rospy.Publisher(disturbance_update_topic, HiLoArray, queue_size=1)

        # Set up services
        modify_environment_service = rospy.get_param("~services/modify_environment")
        rospy.Service(modify_environment_service, ModifyEnvironment, self.handle_modified_environment)


    def update_disturbances(self):
        if self.disturbance_idx >= len(self.disturbance_update_list):
            rospy.logwarn("No more disturbances to update, no update sent")
        else:
            key = list(self.disturbance_update_list.keys())[self.disturbance_idx]
            disturbance_space = self.disturbance_update_list[key]
            hi = np.array(disturbance_space["hi"])
            lo = np.array(disturbance_space["lo"])
            self.disturbance_idx += 1
            self.disturbance_update_pub.publish(HiLoArray(hi=hi, lo=lo))
    
    def update_actuation(self):
        if self.actuation_idx >= len(self.actuation_update_list):
            rospy.logwarn("No more actuations to update, no update sent")
        else:
            control_space = self.actuation_update_list[self.actuation_idx]
            hi = np.array(control_space["hi"])
            lo = np.array(control_space["lo"])
            self.actuation_idx += 1
            self.actuation_update_pub.publish(HiLoArray(hi=hi, lo=lo))
    
    def handle_modified_environment(self, req):
        '''
        To add disturbances, paste the following in a terminal:
          rosservice call /env/modify_environment "update_disturbance"
          rosservice call /env/modify_environment "update_actuation"
        '''
        modification_request = req.modification
        # if modification_request == "actuation":
        #     # Get the new actuation values
        #     self.actuation_update_pub.publish()
        if modification_request == "update_disturbance":
            self.update_disturbances()
            output = "Disturbance Updated"
        elif modification_request == "update_actuation":
            self.update_actuation()
            output = "Actuation Updated"
        else:
            output = "Invalid modification request"
        return ModifyEnvironmentResponse(output)


if __name__ == "__main__":
    rospy.init_node("modify_environment_node")
    modify_environment_server = ModifyEnvironmentServer()
    rospy.spin()
