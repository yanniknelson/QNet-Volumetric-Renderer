from Bounds import *

class Transform4:
    def __init__(self, trans = None, inv = None):
        if trans is None:
            self.trans = np.identity(4)
        else:
            self.trans = trans
        if inv is None:
            self.inv = np.linalg.inv(self.trans)
        else:
            self.inv = inv

    def inverse(self):
        return Transform4(self.inv, self.trans)

    def T(self):
        return Transform4(self.trans.T, self.inv.T)

    def TransformPoint(self, p):
        tp = np.dot(self.trans, np.concatenate((p, [1])).T)
        return tp[:3]/tp[3]

    def TransformDirection(self, p):
        tp = np.dot(self.trans, np.concatenate((p, [0])).T)
        return tp[:3]

    def TransformNormal(self, p):
        tp = np.dot(self.inv, np.concatenate((p, [0])).T)
        return tp[:3]

    def TransformRay(self, r):
        return Ray(self.TransformPoint(r.o), self.TransformDirection(r.d))

    def __str__(self):
        return "{0}, \n {1}".format(self.trans, self.inv)

    def __mul__(self, t2):
        return Transform4(np.dot(self.trans, t2.trans), np.dot(t2.inv, self.inv))

class Scale(Transform4):
    def __init__(self, x, y, z):
        super().__init__()
        self.trans[0][0] = x
        self.trans[1][1] = y
        self.trans[2][2] = z
        self.inv[0][0] = 1/x
        self.inv[1][1] = 1/y
        self.inv[2][2] = 1/z

class LookAt(Transform4):
    def __init__(self,pos, look, up):
        super().__init__()
        dir = (look - pos)
        dir = dir/np.linalg.norm(dir)
        left = np.cross(up/np.linalg.norm(up), dir)
        left = left/np.linalg.norm(left)
        newUp = np.cross(dir, left)

        # translation
        self.trans[0][3] = pos[0]
        self.trans[1][3] = pos[1]
        self.trans[2][3] = pos[2]
        self.trans[3][3] = 1

        self.trans[0][0] = left[0]
        self.trans[1][0] = left[1]
        self.trans[2][0] = left[2]
        self.trans[3][0] = 0

        self.trans[0][1] = newUp[0]
        self.trans[1][1] = newUp[1]
        self.trans[2][1] = newUp[2]
        self.trans[3][1] = 0

        self.trans[0][2] = dir[0]
        self.trans[1][2] = dir[1]
        self.trans[2][2] = dir[2]
        self.trans[3][2] = 0

        self.inv = np.linalg.inv(self.trans)

class Perspective(Transform4):
    def __init__(self, fov, n, f):
        super().__init__()
        self.trans[2][2] = f/(f-n)
        self.trans[2][3] = -f*n/(f-n)
        self.trans[3][2] = 1
        self.trans[3][3] = 0
        invTanAng = 1/np.tan(2*np.pi * (fov/720))
        t = Scale(invTanAng, invTanAng, 1) * Transform4(self.trans)
        self.trans = t.trans
        self.inv = t.inv

class Translate(Transform4):
    def __init__(self, x, y, z):
        super().__init__()
        self.trans[0][3] = x
        self.trans[1][3] = y
        self.trans[2][3] = z
        self.inv[0][3] = -x
        self.inv[1][3] = -y
        self.inv[2][3] = -z
