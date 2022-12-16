from __future__ import annotations

from itertools import product

import numpy as np
from PIL import Image
import rospy

from utils.Node import Node, OpenList, VisitedList
from utils.utils import vec_norm, manh_dist
from planners.Planner import Mapper


class Astar(Mapper):
    def __init__(self,image_path, scale = 1,neighborhood=8,postprocess=False):
        super().__init__(image_path,postprocess)
        self.scale = scale
        assert neighborhood in [4,8]
        self.neighborhood = neighborhood

    def gen_node(self,coord, add=False):
        neighbors = self._get_neighbors(coord,add)
        return Node(coord,0,0,0,None,neighbors)

    def __call__(self,goal,position):
        self._initialized = False
        rospy.loginfo("Calculating path")
        self._initialized = self.search_path(position,goal)
        rospy.loginfo("Path calculated")
        return self

    def setup_search(self,position,goal):
        self.visited = VisitedList()
        self.openlist = OpenList()
        coord = self.pos2coord(position)
        start = self.gen_node(coord)
        self.openlist.insert(start)
        self.goal = self.pos2coord(goal)

    def search_path(self,position,goal):
        self.setup_search(position,goal)
        node = self.search_loop()
        if not node:
            rospy.loginfo("No path available")
            return False
        self.path = []
        while node != None:
            self.path.insert(0,node.coord)
            node = node.parent
        print(self.path)
        self.save_graph_img()
        return True

    def _get_neighbors(self,coord,add=False):
        x,y = int(coord[0]),int(coord[1])
        l,r = max(0, x-1), min(self.size[0],x+1)
        t,b = max(0, y-1), min(self.size[0],y+1)
        candidates = [(x,t),(x,b),(l,y),(r,y)]
        candidates = [c for c in set(candidates) if c != (x,y) and self.map[c]] 
        if self.neighborhood == 8:
            ax1, ax2 = zip(*candidates)
            ax1, ax2 = [i for i in ax1 if x!=i], [j for j in ax2 if y!=j]
            diag = product(ax1,ax2)
            candidates.extend(c for c in diag if self.map[c])
        return candidates

    def search_loop(self) -> Node|None:
        while self.openlist:
            node = self.openlist.pop() # Best heuristic + cost
            if node.coord == self.goal:
                return node
            self.visited.add(node.coord)
            child_nodes = node.neighbors # Get children and parent
            child_nodes = [n for n in child_nodes if n not in self.visited]
            self.add2open(node,child_nodes) # Add to list handled in OpenList
        return None

    def add2open(self, parent:Node, children):
        for coord in children:
            if ( isinstance(coord[0], int) and
                 isinstance(coord[1], int) and
                 self.map[coord] == 0 ):
                self.visited.add(coord)
            node_cost = np.sqrt(manh_dist(parent.coord,coord))
            depth = parent.depth + 1
            heuristic = vec_norm(coord, self.goal)[1]
            real_cost = node_cost+parent.real_cost
            neighbors = self._get_neighbors(coord)
            n = Node(coord,real_cost,heuristic,depth,parent,neighbors)
            self.openlist.insert(n)
