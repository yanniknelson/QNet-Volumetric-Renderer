import numpy as np

class Ray:
    def __init__(self, o = None, dir = None):
        if o is None:
            self.o = np.array([0,0,0])
        else:
            self.o = o
        if dir is None:
            self.d = np.array([0,0,0])
        else:
            self.d = dir

    def __str__(self):
        return "o = {0}, d = {1}".format(self.o, self.d)