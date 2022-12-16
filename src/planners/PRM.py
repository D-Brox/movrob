from __future__ import annotations

import numpy as np
from numpy.random import uniform as random_range
import matplotlib.pyplot as plt

from planners.Astar import Astar
from tqdm import tqdm

from utils.Node import OpenList, VisitedList
from utils.utils import vec_norm

class PRM(Astar):
    def __init__(self,image_path, scale = 0.125, neighborhood=4, n=300, postprocess=True):
        super().__init__(image_path, scale,neighborhood,postprocess)
        self.gen_random_nodes(n)

    def save_graph_img(self):
        self.fig = plt.figure()
        img = plt.imread(self.image_path)
        plt.imshow(img)
        for i,(k,v) in tqdm(enumerate(self.graph.items()),total = len(self.graph)):
            for n in v.neighbors:
                if list(self.graph.keys()).index(n) < i:
                    plt.plot([k[0],n[0]],[k[1],n[1]],'ob', linestyle="-",lw=0.1,markersize=0)
        for k,v in tqdm(self.graph.items()):
            plt.plot(k[0], k[1], 'or',markersize=2)
        for k,n in tqdm(zip(self.path[:-1],self.path[1:])):
            plt.plot([k[0],n[0]],[k[1],n[1]],'og', linestyle="-",lw=2,markersize=0)
            plt.plot(k[0], k[1], 'og',markersize=5)
            plt.plot(n[0], n[1], 'og',markersize=5)
        plt.show()
        plt.savefig("./teste.png")

    def gen_random_nodes(self,n):
        tq = tqdm(total = n)
        while len(self.graph) < n:
            x = random_range(0,self.size[0])
            y = random_range(0,self.size[1])
            coord = tuple(np.int64((x,y)))
            if self.map[coord]:
                self.graph[(x,y)] = self.gen_node((x,y))
                tq.update()

    def _get_neighbors(self,coord,add=False):
        if coord in self.graph:
            return self.graph[coord].neighbors
        else:
            neighbors = []
            for node_coord in self.graph:
                if (self.postprocess and vec_norm(coord,node_coord)[1] > 80) \
                   or self.check_collision(coord,node_coord):
                    continue
                neighbors.append(node_coord)
                if add:
                    self.graph[node_coord].neighbors.append(coord)
            return neighbors

    def setup_search(self, position, goal):
        self.visited = VisitedList()
        self.openlist = OpenList()
        coord = self.pos2coord(position)

        post = self.postprocess
        self.postprocess = False
        self.graph[coord] = self.gen_node(coord, add=True)
        self.openlist.insert(self.graph[coord])

        self.goal = self.pos2coord(goal)
        self.graph[self.goal] = self.gen_node(self.goal, add=True)
        self.postprocess = post

