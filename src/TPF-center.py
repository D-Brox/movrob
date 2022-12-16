#!/usr/bin/env python3
import rospy
from message_filters import Subscriber, TimeSynchronizer
from geometry_msgs.msg import Point, Twist, Pose
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

import numpy as np

from controllers.Controller import Node
from controllers.DiferentialRobot import DifferentialRobot
from planners.Swarm import CenterSwarm

class SwarmNode(Node):
    def __init__(self,mode,n_robots=10,n_classes=4):
        self.dt = 1/60
        super().__init__(freq=1/self.dt)
        self.controller = DifferentialRobot(0.01)
        self.pub_vel = []
        self.sub = []
        self.robots = n_robots*n_classes
        self.classes = n_classes
        self.planner = CenterSwarm(mode,n_robots,n_classes)

        for i in range(self.robots):
            self.pub_vel.append(rospy.Publisher(f"/robot_{i}/cmd_vel", Twist, queue_size=10))
            self.sub.append(Subscriber(f"/robot_{i}/base_pose_ground_truth",Odometry))
        self.multi_sub = TimeSynchronizer(self.sub,self.robots)
        self.multi_sub.registerCallback(self.callback_all)

    def callback_all(self,*d):
        theta = [None]*len(d)
        for i in range(len(d)):
            theta[i] = self.callback_pose(d[i],i)
        V = [self.planner.get_next(i,self.dt) for i in range(len(d))]
        self.callback_vel(V,theta)
        self.rate.sleep()

    def callback_pose(self,d,i):
        pos, theta = self.get_pose(d.pose.pose)
        self.planner.update_position(i,pos)
        return theta

    def callback_vel(self,U,theta):
        U = np.array(U)
        U -= np.mean(U,axis = 0)

        for i,(u,t) in enumerate(zip(U,theta)):
            vel = self.controller.feedback_linearization(u,t)
            self.pub_vel[i].publish(vel)

    def get_pose (self,data):
        x = data.position.x
        y = data.position.y
        quat = data.orientation
        quat = [quat.x,quat.y,quat.z,quat.w]
        _,_,theta = euler_from_quaternion(quat)
        return (x,y),theta

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        image_path = "/workspaces/src/movrob/worlds/test_tangent.png"
        node = SwarmNode(mode = "aggregate")
        node.run()
    except rospy.ROSInterruptException:
        pass