from Transforms import *

class Camera:
    def __init__(self, height=None, width=None, fov=None, pos=None, up=None, lookat=None):
        if (height is None):
            return
        asprat = width/height
        if (asprat > 1):
            pMin = [-asprat, -1]
            pMax = [asprat, 1]
        else:
            pMin = [-1, -1/asprat]
            pMax = [1, 1/asprat]

        self.CameraToWorld = LookAt(pos, lookat, up)
        CameraToScreen = Perspective(fov, 1e-2, 1000)
        ScreenToRaster = Scale(width, height, 1) * \
                Scale(1/(pMax[0] - pMin[0]), 1/(pMin[1] - pMax[1]), 1) * \
                Translate(-pMin[0], -pMax[1], 0)
        RasterToScreen = ScreenToRaster.inverse()
        self.RasterToCamera = CameraToScreen.inverse() * RasterToScreen

    def GenerateRay(self, x, y):
        pCamera = self.RasterToCamera.TransformPoint(np.array([x,y,0]))
        pCamera = pCamera/np.linalg.norm(pCamera)
        return self.CameraToWorld.TransformRay(Ray(np.array([0,0,0]), pCamera))