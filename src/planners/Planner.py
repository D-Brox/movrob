from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

import rospy

from utils.utils import line_grid_collision, vec_norm

class Planner(ABC):
    def __init__():
        return

    @abstractmethod
    def get_next(self, goal, position, angle):
        return

class Mapper(Planner):
    def __init__(self,image_path, postprocess=False):
        self.image = Image.open(image_path).convert('1')
        self.map = np.transpose(np.array(self.image))
        self.size = np.array(self.image.size)
        self.scale = 1
        self.image_path = image_path
        self._initialized = False
        self.postprocess = postprocess
        self.goal = None
        self.path = []
        self.graph = {}

    def pos2coord(self,position,round = True):
        coord = ((self.size[0]-1)/2 + position[0]/self.scale,
                 (self.size[1]-1)/2 - position[1]/self.scale)
        if round:
            return tuple(int(c) for c in coord)
        else:
            return coord
    
    def check_collision(self,p1,p2):
        return line_grid_collision(p1,p2,self.map,d=0.2/self.scale)
    
    def get_next(self, position, goal):
        if not self._initialized:
            return None
        coord = self.pos2coord(position,round = False)
        if not self.path:
            vec, norm = vec_norm(goal,position)
            if norm < 0.2:
                rospy.loginfo("Goal reached.")
                self._initialized = False
                return None
        else:
            mid_goal = self.path[0]
            vec,norm =  vec_norm(mid_goal,coord)
            if norm < 0.2 or (len(self.path) > 1 and not
               self.check_collision(self.path[1],coord)):
                self.path.pop(0)
                return (0,0)
        U = vec/norm
        if self.path:
            U[1] = -U[1]
        return U

    def save_graph_img(self):
        return
