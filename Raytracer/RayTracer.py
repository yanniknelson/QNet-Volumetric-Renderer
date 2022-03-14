from alive_progress import alive_bar
from Qnet import *
from Marcher import Marcher
from Camera import *
import timeit



np.seterr(divide='ignore')

width = 400
height = 400

pos = np.array([4, 0, 0])
up = np.array([0,0,1])
lookat = np.array([0,0,0])

image = np.zeros((height, width))
reference = np.zeros((height, width))

marcher = Marcher(np.array([-1,-1,-1]), np.array([1,1,1]), "../fluid_data_0083_numpy_array.npy")

qnet_time = 0
voxel_time = 0

with torch.no_grad():
    weights = scio.loadmat("../MATLABtest/volume_weights_v2.mat")

    qnet = Intergrator(weights["pw1"], weights["pb1"], weights["pw2"], weights["pb2"], False, weights["yoffset"], weights["ymin"], weights["yrange"])

    c = Camera(height, width, 35, pos, up, lookat)
    vol = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))


    print("qnet start")
    # with alive_bar(width * height) as bar:
    start_time = timeit.default_timer()
    for y in range(height):
        for x in range(width):
            ray = c.GenerateRay(x,y)
            hit, t0, t1 = vol.intersect(ray)
            if hit:
                intersectionPoint = ray.o + t0* ray.d
                # image[y].append(marcher.trace_scaling(intersectionPoint, ray.d))
                image[y][x] = qnet.IntegrateRay(ray, t0, t1)
                # bar()
    end_time = timeit.default_timer()

qnet_time = end_time - start_time

print("Qnet render finished in: ", qnet_time, "\nvoxel start")

start_time = timeit.default_timer()
for y in range(height):
    for x in range(width):
        ray = c.GenerateRay(x,y)
        hit, t0, t1 = vol.intersect(ray)
        if hit:
            intersectionPoint = ray.o + t0* ray.d
            reference[y][x] = marcher.trace_scaling(intersectionPoint, ray.d)
end_time = timeit.default_timer()

voxel_time = end_time - start_time

fig, axes = plt.subplots(1,2)
print("qnet render time = ", qnet_time)
print("voxel render time = ", voxel_time)
RMSE = np.sqrt(np.mean((reference-image)**2))
print("RMSE = ", RMSE)
im = axes[0].imshow(image)
fig.colorbar(im, ax=axes[0])
rf = axes[1].imshow(reference)
fig.colorbar(rf, ax=axes[1])
relativeError = np.mean(np.divide(np.abs(image-reference),(reference + 0.1)))
print("RE = ", relativeError, flush=True)
plt.show()
