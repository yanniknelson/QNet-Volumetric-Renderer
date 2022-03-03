from Ray import *

class Bounds3:
    def __init__(self, p1 = None, p2 = None):
        if (p1 is None and p2 is None):
            print("creating")
            self.pMin = np.array([float('inf'),float('inf'),float('inf')])
            self.pMax = np.array([float('-inf'),float('-inf'),float('-inf')])
            return
        
        if p2 is None:
            self.pMin = p1
            self.pMax = p1
        else:
            self.pMin = np.array([min(p1[0], p2[0]),min(p1[1], p2[1]),min(p1[2], p2[2])])
            self.pMax = np.array([max(p1[0], p2[0]),max(p1[1], p2[1]),max(p1[2], p2[2])])

    def intersect(self, ray):
        t0 = 0
        t1 = float('inf')
        for i in range(3):
            invRayDir = 1/ray.d[i]
            tNear = (self.pMin[i] - ray.o[i]) * invRayDir
            tFar = (self.pMax[i] - ray.o[i]) * invRayDir
            if (tNear > tFar):
                tNear, tFar = tFar, tNear
                tFar *= 1 + 2 * ((3* np.finfo(float).eps)/(1-3* np.finfo(float).eps))
            if tNear > t0: 
                t0 = tNear
            if tFar < t1: 
                t1 = tFar
            if (t0 > t1): 
                return False, t0, t1
        return True, t0, t1