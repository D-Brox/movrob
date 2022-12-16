from abc import ABC, abstractmethod
import rospy

from geometry_msgs.msg import Point, Twist, Pose
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

#from src.LidarScanner import LidarScanner
from controllers.DiferentialRobot import DifferentialRobot

class Node(ABC):
    def __init__(self,freq=30.0):
        rospy.init_node('control_node')
                
        # Set time rate
        rospy.Subscriber('/clock', Clock, self.callback_time)
        self.freq = freq
        self.rate = rospy.Rate(self.freq)
        self.rate.sleep()

    def callback_time(self, data):
        self.time = data.clock.secs*1e3 + data.clock.nsecs/1e6


class ControlNode(Node):
    def __init__(self,freq=10.0):
        super().__init__(freq)

        self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Set current goal
        self.goal = None
        self.U = (0,0,0)
        self.vel = Twist()
        rospy.Subscriber('/setup_goal',Point,self.callback_goal)

        #Set current position
        self.controller = DifferentialRobot(0.2)
        self.x = None
        self.y = None
        self.theta = 0
        rospy.Subscriber("/base_pose_ground_truth",Odometry, self.callback_pose_odometry) #
        rospy.Subscriber("/setup_origin",Pose,self.callback_pose)
        self.rate.sleep()

    ###############
    # ROS updates #
    ###############

    def callback_goal(self,data):
        old_goal = self.goal
        new_goal = (data.x,data.y)
        while (None in [self.x,self.y]):
            self.rate.sleep()
        if old_goal != new_goal:
            rospy.loginfo(f"Goal set to {new_goal}")
            self.goal = new_goal
            self.goal_update()
            self.rate.sleep()

    def goal_update(self):
        # WaveFront and other methods that need extra computation
        return

    def callback_pose_odometry(self,data):
        self.callback_pose(data.pose.pose)

    def callback_pose (self,data):
        self.x = data.position.x
        self.y = data.position.y
        quat = data.orientation
        quat = [quat.x,quat.y,quat.z,quat.w]
        _,_,self.theta = euler_from_quaternion(quat)

    def publish_vel(self):
        self.pub_vel.publish(self.vel)

    ####################
    # Robot controller #
    ####################
    
    def startup(self):
        while self.goal is None or self.x is None or self.y is None:
            rospy.loginfo_once("Waiting for goal to be set")
            self.rate.sleep()

    def setup(self):
        return
    
    @abstractmethod
    def plan():
        return

    def run(self):
        self.startup()
        
        self.setup()
        
        while not rospy.is_shutdown():
            self.plan()
            if self.U is not None:
                self.vel = self.controller.feedback_linearization(self.U,self.theta)
                self.publish_vel()
            self.rate.sleep()
