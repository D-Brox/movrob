from __future__ import annotations

import numpy as np
from numpy.random import uniform as random_range
from tqdm import tqdm
import matplotlib.pyplot as plt

import rospy

from utils.Node import Node
from utils.utils import find_closest, vec_norm
from planners.Planner import Mapper

class RRT(Mapper):
    def __init__(self, image_path, scale=0.125, step=128, postprocess=False):
        super().__init__(image_path, postprocess)
        self.scale = scale
        self.step = step
        
        self.image_path = image_path

    def __call__(self,goal,position):
        if self.goal == goal:
            return self
        self._initialized = False
        rospy.loginfo("Calculating path")
        self.search_path(position,goal)
        self._initialized = True
        rospy.loginfo("Path calculated")

        return self
    
    def search_path(self,position,goal):
        coord = self.pos2coord(position,round=False)
        self.goal = self.pos2coord(goal,round=False)
        self.graph[coord] = Node(coord,0,0,0,None,None)
        while True:
            n = self.gen_random_point()
            v = self.closest_v(n)
            vec,norm = vec_norm(n,v)
            p = tuple(v + self.step*vec/norm)
            v = self.closest_v(p)

            if p[0] >= self.size[0] or p[1] >= self.size[1] or p[0] < 0 or p[1] < 0 \
               or not self.map[int(p[0]),int(p[1])] or self.check_collision(v,p):
                continue
            
            node = self.graph[v]
            self.graph[p] = Node(p,0,0,node.depth+1,node,None)
            if not self.check_collision(p,self.goal):
                vert = self.graph[p]
                self.graph[self.goal] = Node(self.goal,0,0,vert.depth+1,vert,None)
                break
        self.path = []
        node = self.graph[self.goal]
        while node:
            self.path.insert(0,node.coord)
            node = node.parent
        print(self.path)
        self.save_graph_img()

    def save_graph_img(self):
        img = plt.imread(self.image_path)
        plt.imshow(img)
        for k,v in tqdm(self.graph.items()):
            if v.parent:
                n = v.parent.coord
                plt.plot([k[0],n[0]],[k[1],n[1]],'ob', linestyle="-",lw=0.2,markersize=1)
                plt.plot(k[0], k[1], 'or',markersize=2)
                plt.plot(n[0], n[1], 'or',markersize=2)
        for k,n in tqdm(zip(self.path[:-1],self.path[1:])):
            plt.plot([k[0],n[0]],[k[1],n[1]], 'og', linestyle="-",lw=1,markersize=1)
            plt.plot(k[0], k[1], 'og',markersize=2)
            plt.plot(n[0], n[1], 'og',markersize=2)
        plt.show()
        plt.savefig("./teste.png")


    def gen_random_point(self):
        x = random_range(0,self.size[0])
        y = random_range(0,self.size[1])
        return x,y
    
    def closest_v(self,n):
        return tuple(find_closest(n,np.array(list(self.graph.keys()))))