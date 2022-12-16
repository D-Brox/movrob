
import numpy as np

import rospy

from planners.Planner import Planner
from utils.utils import check_nan, vec_norm

from numba import njit, prange

@njit(cache=True)
def grad_u_ij(alpha,dij,qi,qj):
    qij = (qi - qj)
    n_qij_2 = qij[0]**2+qij[1]**2
    n_qij = np.sqrt(n_qij_2)
    scalar = alpha*(n_qij-dij+1/n_qij-dij/(n_qij_2))
    return scalar*qij/n_qij

@njit(parallel=True,cache=True)
def get_ui(i,v,q,alpha,Daa,Dab,N,M):
    u = np.zeros((N,2),dtype=float)
    qi = q[i]
    c = i%M
    vi = v[i]
    for j in prange(N):
        if i == j:
            continue
        if j%M == c:
            dij = Daa[c]
        else:
            dij = Dab
        u[j] = grad_u_ij(alpha,dij,qi,q[j])
    return -np.sum(u,axis=0)+np.sum(v,axis=0)-vi*N

class Swarm(Planner):
    def __init__(self,mode, n_robots, n_classes):
        self.robots = n_robots*n_classes
        self.classes = n_classes
        self.gen_d(mode)
        self.v = np.zeros((self.robots,2),dtype=float)
        self.initialized = False
        self.pos = np.tile(np.nan,(self.robots,2))

    def gen_d(self,mode):
        if mode == "cluster":
            self.alpha = 0.2
            self.Daa = np.repeat(0.25,self.classes)
            self.Dab = 0.5
        elif mode == "radial":
            self.alpha = 2.2
            self.Daa = 0.20+0.5*(np.arange(0,self.classes))
            self.Dab = 0.4
        elif mode == "aggregate":
            self.alpha = 0.2
            self.Daa = np.repeat(0.5,self.classes)
            self.Dab = 0.2
        elif mode == "hybrid":
            assert self.classes == 4
            self.alpha = 0.2
            self.Daa = np.array([0.2,0.2,0.6,0.8])
            self.Dab = 0.3

    def update_position(self, i, pose):
        self.pos[i] = pose
        if (not self.initialized) and (not check_nan(self.pos)):
            self.initialized = True

    def get_next(self, i,dt):
        if not self.initialized:
            return (0,0)
        self.v[i] += get_ui(i,self.v,self.pos,self.alpha,
                            self.Daa,self.Dab,
                            self.robots,self.classes) * dt
        return tuple(self.v[i])

class CenterSwarm(Swarm):
    def gen_d(self, mode):
        if mode == "cluster":
            self.alpha = 1
            self.Daa = np.repeat(0.25,self.classes)
            self.Dab = 0.5
        elif mode == "radial":
            self.alpha = 1.5
            self.Daa = 0.20+0.9*(np.arange(0,self.classes))
            self.Dab = 1
        elif mode == "aggregate":
            self.alpha = 0.5
            self.Daa = np.repeat(0.5,self.classes)
            self.Dab = 0.2
        elif mode == "hybrid":
            assert self.classes == 4
            self.alpha = 0.6
            self.Daa = np.array([0.2,0.2,0.6,0.6])
            self.Dab = 0.3


@njit(parallel=True,cache=True)
def get_ui_dab(i,v,q,alpha,Dab,N,M):
    u = np.zeros((N,2),dtype=float)
    qi = q[i]
    c = i%M
    vi = v[i]
    for j in prange(N):
        if i == j:
            continue
        d = j%M
        dij = Dab[c][d]
        qj = q[j]
        u[j] = grad_u_ij(alpha,dij,qi,qj)
    return -np.sum(u,axis=0)+np.sum(v,axis=0)-vi*N

class DabSwarm(CenterSwarm):
    def gen_d(self, mode):
        assert self.classes == 4
        self.alpha = 0.7
        self.Dab = np.array([[0.2 , 0.75, 0.35, 0.55],  # C0 se aproxima de C2 e se afasta de C1
                             [0.35, 0.2 , 0.75, 0.55],  # C1 se aproxima de C0 e se afasta de C2
                             [0.75, 0.35, 0.2 , 0.55],  # C2 se aproxima de C1 e se afasta de C0
                             [0.35, 0.35, 0.35, 2   ]]) # C3 fica envolvendo

    def get_next(self, i,dt):
        if not self.initialized:
            return (0,0)
        self.v[i] += get_ui_dab(i,self.v,self.pos,
                                self.alpha,self.Dab,
                                self.robots,self.classes) * dt
        return tuple(self.v[i])