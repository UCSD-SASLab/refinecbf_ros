#!/usr/bin/env python3

import rospy
from refinecbf_ros.msg import Array, HiLoArray, ValueFunctionMsg
from refinecbf_ros.srv import ModifyEnvironment

class ModifyEnvironmentServer:
    def __init__(self):
        # Set up publishers
        self.actuation_update_pub = rospy.Publisher("/actuation_update", HiLoArray, queue_size=1)
        self.disturbance_update_pub = rospy.Publisher("/disturbance_update", HiLoArray, queue_size=1)
        self.obstacle_update_pub = rospy.Publisher("/obstacle_update", ValueFunctionMsg, queue_size=1)

        # Set up service
        self.service = rospy.Service("modify_environment", ModifyEnvironment, self.handle_modified_environment)

    def handle_modified_environment(self, req):
        modification_request = req.modification
        if modification_request == "actuation":
            # Get the new actuation values
            self.actuation_update_pub.p