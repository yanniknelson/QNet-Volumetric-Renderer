from numpy import arange
from Qnet import *
from Marcher import Marcher
from Camera import *
import timeit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pathlib import Path

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label('Optical Depth', rotation=270, verticalalignment='baseline')
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.sca(last_axes)
    return cbar

np.seterr(divide='ignore')

width = 100
height = 100

pos = np.array([4, 0, 0])
up = np.array([0,0,1])
lookat = np.array([0,0,0])

image = np.zeros((height, width))
reference = np.zeros((height, width))

reference_data = "Blender_cloud"
net_version = 1

marcher = Marcher(np.array([-1,-1,-1]), np.array([1,1,1]), f"../volumes/npversions/{reference_data}.npy")

qnet_time = 0
voxel_time = 0

for gpu in [False, True]:
    GPU = "GPU" if gpu else "No_GPU"
    with torch.no_grad():
        weights = scio.loadmat(f"../MATLABtest/{reference_data}_weights_v{net_version}.mat")
        qnet = Intergrator(weights["pw1"], weights["pb1"], weights["pw2"], weights["pb2"], False, weights["yoffset"], weights["ymin"], weights["yrange"], gpu)

    vol = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))

    results = []
    for (width, height) in [(x,x) for x in [10, 20, 40, 80, 160, 320, 640]]:
        print("qnet start ", width, height)
        c = Camera(height, width, 35, pos, up, lookat)
        image = np.zeros((height, width))
        with torch.no_grad():
            start_time = timeit.default_timer()
            for y in range(height):
                for x in range(width):
                    ray = c.GenerateRay(x,y)
                    hit, t0, t1 = vol.intersect(ray)
                    if hit:
                        image[y][x] = qnet.IntegrateRay(ray, t0, t1)

            end_time = timeit.default_timer()
        qnet_time = end_time - start_time
        results.append(qnet_time)
        print("Qnet render finished in: ", qnet_time)
        fig, axes = plt.subplots(1,1)
        axes.axis('off')
        axes.title.set_text("Q-Net")
        im = axes.imshow(np.array(image))
        colorbar(im)
        plt.tight_layout()
        if not os.path.exists(f'../Renders/GPUExperiments/{GPU}/Renders'):
            if not os.path.exists(f'../Renders/GPUExperiments/{GPU}'):
                if not os.path.exists(f'../Renders/GPUExperiments/'):
                    os.makedirs(f'../Renders/GPUExperiments/')
                os.makedirs(f'../Renders/GPUExperiments/{GPU}')
            os.makedirs(f'../Renders/GPUExperiments/{GPU}/Renders')
        plt.savefig(f"../Renders/GPUExperiments/{GPU}/Renders/{reference_data}_v{net_version}_{width}_{height}_{pos[0]}_{pos[1]}_{pos[2]}.png")
        fle = Path(f"../Renders/GPUExperiments/{GPU}/data.txt")
        fle.touch(exist_ok=True)
        f = open(f"../Renders/GPUExperiments/{GPU}/data.txt", 'a')
        f.write(f"{width},{qnet_time}\n")
        f.close()



# start_time = timeit.default_timer()
# for y in range(height):
#     for x in range(width):
#         ray = c.GenerateRay(x,y)
#         hit, t0, t1 = vol.intersect(ray)
#         if hit:
#             reference[y][x] = marcher.trace_scaling(ray.o + ray.d*t0, ray.d)
            
            
# end_time = timeit.default_timer()

# voxel_time = end_time - start_time

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
