import rospy
from src.Controller import ControlNode
from src.TangentBug import TangentBug, TB_state
from src.LidarScanner import LidarScanner

class TangentNode(ControlNode):
    def __init__(self,params):
        super().__init__(params)
        self.robot = TangentBug
        self.lidar = LidarScanner('/base_scan')
    
    def iteration(self):
        if self.robot.state == TB_state.FOLLOW_GOAL:
            pass
        elif self.robot.state == TB_state.FOLLOW_Oi:
            pass
        elif self.robot.state == TB_state.FOLLOW_TANG:
            pass
        elif self.robot.state == TB_state.REACHED_GOAL:
            pass

    def target(self):
        pass


if __name__ == "__main__":
    try:
        params = {"x_goal":0,
                  "y_goal":0}
        node = TangentNode(params)
    except rospy.ROSInterruptException:
        pass