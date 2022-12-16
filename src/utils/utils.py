import numpy as np
from numba import njit, prange

@njit
def vec_norm(p1,p2):
    vec = np.array(p1) - np.array(p2)
    return vec, np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def manh_dist(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    vec = p1-p2
    return np.linalg.norm(vec,1)

def get_measured_points(x,y,theta,d):
    x = x + d* np.cos(theta)
    y = y + d* np.sin(theta)
    return (x,y)

def idxmin(l):
    return np.argmin(l)

def relative_angle(p1,p2,theta):
    v2, norm = vec_norm(p2,p1)
    u1 = np.array([1,0])
    u2 = v2/norm
    phi = np.arccos(np.clip(np.dot(u2, u1), -1.0, 1.0)) #* ccw([0,0],u1,u2)
    angle = phi - theta
    while angle >=np.pi:
        angle -= np.pi
    while angle >=np.pi:
        angle += np.pi
    return angle

# Check if counterclockwise
@njit
def ccw(A,B,C):
    cc = (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return 1 if cc else -1

def ortogonal_vec(v):
    return np.array([-v[1],v[0]])

# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def segment_dist(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    A = np.array(A)
    B = np.array(B)
    P = np.array(P)
    if np.all(A == P) or np.all(B == P):
        return 0
    if np.arccos(np.dot((P - A) / np.linalg.norm(P - A), (B - A) / np.linalg.norm(B - A))) > np.pi / 2:
        return np.linalg.norm(P - A)
    if np.arccos(np.dot((P - B) / np.linalg.norm(P - B), (A - B) / np.linalg.norm(A - B))) > np.pi / 2:
        return np.linalg.norm(P - B)
    return np.linalg.norm(np.cross(A-B, A-P))/np.linalg.norm(B - A)

@njit
def line_grid_collision(A,B,map,d=0.2):
    x1,x2 = min(int(A[0]),int(B[0])),max(int(A[0]),int(B[0]))
    y1,y2 = min(int(A[1]),int(B[1])),max(int(A[1]),int(B[1]))

    for i in prange(x1,x2+1):
        for j in prange(y1,y2+1):
            if map[i,j]:
                continue
            corners = [(i-d,j-d),(i+d+1,j-d),(i-d,j+d+1),(i+d+1,j+d+1)]
            clock = 0
            for p in corners:
                if clock == 0:
                    clock = ccw(A,B,p)
                else:
                    if clock != ccw(A,B,p):
                        return True
    return False

@njit
def find_closest(p,graph):
    dist = np.inf
    vert = None
    for v in graph:
        d = np.sqrt((p[0]-v[0])**2 + (p[1]-v[1])**2)
        if d < dist:
            dist = d 
            vert = v
    return vert

@njit
def check_nan(m):
    return np.isnan(m).any()

