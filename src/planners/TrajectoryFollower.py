import numpy as np

import rospy
from rosgraph_msgs.msg import Clock

from planners.Planner import Planner

class TrajectoryFollower(Planner):
    def __init__(self,path,dt=0.000001):
        self.path = path
        self.dt = dt
        rospy.Subscriber('/clock', Clock, self.callback_time)
    def startup(self):
        while self.x is None or self.y is None:
            rospy.loginfo_once("Waiting for goal to be set")
            self.rate.sleep()
    def callback_time(self,data):
        self.time = data.clock.secs*1e3 + data.clock.nsecs/1e6

    def get_next(self,position):
        pose = np.array(position)
        target = self.path(self.time)
        err_p = target - pose
        ff = (self.path(self.time+self.dt)-self.path(self.time-self.dt))/(2*self.dt)
        U = err_p + ff
        return U