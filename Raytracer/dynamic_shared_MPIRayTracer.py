import os
from pathlib import Path
import matplotlib.pyplot as plt
# from matplotlib.pyplot import axes
from mpi4py import MPI 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpi4py.util import dtlib
from Qnet import *
from Marcher import Marcher
from Camera import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # get your process ID

np.seterr(divide='ignore')

width = 400
height = 400

net_version = "1"
reference_data = "Blender_cloud"

exp = "z"

total = width * height


up = np.array([0,0,1])
lookat = np.array([0,0,0])

start_time = None

batchsize = 10

#create image shared memory
dtype = MPI.DOUBLE
np_dtype = dtlib.to_numpy_dtype(dtype)
image_win_size = ((dtype.Get_size() * total) if (rank == 0) else (0))
image_win = MPI.Win.Allocate_shared(image_win_size, dtype.Get_size(), comm=comm)
buf, itemsize = image_win.Shared_query(0) 
image = np.ndarray(buffer=buf, dtype='d', shape=(height,width))

#create reference image shared memory
ref_win = MPI.Win.Allocate_shared(image_win_size, dtype.Get_size(), comm=comm)
buf, itemsize = ref_win.Shared_query(0) 
ref = np.ndarray(buffer=buf, dtype='d', shape=(height,width))

#create shared counter
dtype = MPI.INTEGER
np_dtype = dtlib.to_numpy_dtype(dtype)
counter_win_size = (dtype.Get_size() if (rank == 0) else (0))
counter_win = MPI.Win.Allocate_shared(image_win_size, dtype.Get_size(), comm=comm)
buf, itemsize = counter_win.Shared_query(0) 
counter = np.ndarray(buffer=buf, dtype='d', shape=(1,))

with torch.no_grad():
    if rank == 0:
        weights = scio.loadmat(f"../MATLABtest/{reference_data}_weights_v{net_version}.mat")

        qnet = Intergrator(weights["pw1"], weights["pb1"], weights["pw2"], weights["pb2"], False, weights["yoffset"], weights["ymin"], weights["yrange"])

        marcher = Marcher(np.array([-1,-1,-1]), np.array([1,1,1]), f"../volumes/npversions/{reference_data}.npy")
    else:
        marcher = Marcher()
        qnet = Intergrator()

qnet = comm.bcast(qnet, root=0)
marcher = comm.bcast(marcher, root=0)

start = 215
end = 260

for angle in np.linspace(start, end, int((end-start)//2.5)+1):
    if rank == 0:
        counter[0] = batchsize * size

    start = rank * batchsize

    qnet_time = 0
    voxel_time = 0

    comm.Barrier()

    # pos = [radius, latitude, longitude]
    angularpos = np.array([4, angle, 90])

    phi = angularpos[2]*np.pi/180
    theta = angularpos[1]*np.pi/180

    pos = np.array([angularpos[0]*np.sin(phi)*np.cos(theta),angularpos[0]*np.sin(phi)*np.sin(theta),angularpos[0]*np.cos(phi)])
    if rank == 0:
        print(angle)
        print("camera pos = ", pos, flush=True)

    with torch.no_grad():

        if rank == 0:
            c = Camera(height, width, 35, pos, up, lookat)
        else:
            c = Camera()
        
        c = comm.bcast(c, root=0)

        vol = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))

        go_again = True

        if rank == 0:
            print("qnet start", flush=True)

        comm.Barrier()

        if rank == 0:
            start_time = MPI.Wtime()

        while go_again:
            for i in range(start, min(start + batchsize, total)):
                y = i//width
                x = i%height
                ray = c.GenerateRay(x,y)
                hit, t0, t1 = vol.intersect(ray)
                if hit:
                    image[y][x] = qnet.IntegrateRay(ray, t0, t1)

            go_again = False

            counter_win.Wait()
            counter_win.Lock_all()
            if counter[0] < total:
                go_again = True
                start = int(counter[0])
                counter[0] = min(start + batchsize, total)
            counter_win.Unlock_all()

    comm.Barrier()

    if rank == 0:
        end_time = MPI.Wtime()
        qnet_time = end_time - start_time
        print("Qnet render finished", flush=True)

    go_again = True

    if rank == 0:
        counter[0] = batchsize * size

    start = rank * batchsize

    comm.Barrier()

    if rank == 0:
            start_time = MPI.Wtime()

    while go_again:
        for i in range(start, int((start+batchsize)%(total+1))):
            y = i//width
            x = i%height
            ray = c.GenerateRay(x,y)
            hit, t0, t1 = vol.intersect(ray)
            if hit:
                ref[y][x] = marcher.trace_scaling(ray.o + t0* ray.d, ray.d)

        go_again = False

        counter_win.Wait()
        counter_win.Lock_all()
        if counter[0] < total:
            go_again = True
            start = int(counter[0])
            counter[0] = min(start + batchsize, total)
        counter_win.Unlock_all()

    comm.Barrier()

    if rank == 0:
        end_time = MPI.Wtime()
        voxel_time = end_time - start_time

    if rank == 0:
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
        
        fig, axes = plt.subplots(1,2)
        axes[0].axis('off')
        axes[1].axis('off')
        print("qnet render time = ", qnet_time, flush=True)
        print("voxel render time = ", voxel_time, flush=True)
        RMSE = np.sqrt(np.mean((ref-image)**2))
        print("RMSE = ", RMSE, flush=True)
        relativeError01 = np.divide(np.abs(image-ref),(ref + 0.01))
        RE01 = np.mean(relativeError01)
        RESTD01 = np.std(relativeError01, dtype=np.float64)
        relativeError05 = np.divide(np.abs(image-ref),(ref + 0.05))
        RE05 = np.mean(relativeError05)
        RESTD05 = np.std(relativeError05, dtype=np.float64)
        relativeError1 = np.divide(np.abs(image-ref),(ref + 0.1))
        RE1 = np.mean(relativeError1)
        RESTD1 = np.std(relativeError1, dtype=np.float64)
        relativeError15 = np.divide(np.abs(image-ref),(ref + 0.15))
        RE15 = np.mean(relativeError15)
        RESTD15 = np.std(relativeError15, dtype=np.float64)
        axes[0].title.set_text("Q-Net")
        mn = min(np.min(image), np.min(ref))
        mx = min(np.max(image), np.max(ref))
        im = axes[0].imshow(np.array(image), vmin=mn, vmax=mx)
        colorbar(im)
        axes[1].title.set_text("Ray Marcher")
        rf = axes[1].imshow(np.array(ref), vmin=mn, vmax=mx)
        colorbar(rf)
        plt.tight_layout()
        if not os.path.exists(f'../Renders/{reference_data}_v{net_version}_{exp}_exp_{width}_{height}/'):
            os.makedirs(f'../Renders/{reference_data}_v{net_version}_{exp}_exp_{width}_{height}/')
            os.makedirs(f'../Renders/{reference_data}_v{net_version}_{exp}_exp_{width}_{height}/Plots')
        plt.savefig(f"../Renders/{reference_data}_v{net_version}_{exp}_exp_{width}_{height}/Plots/{reference_data}_v{net_version}_{width}_{height}_{angularpos[0]}_{angularpos[1]}_{angularpos[2]}.png")
        fle = Path(f"../Renders/{reference_data}_v{net_version}_{exp}_exp_{width}_{height}/data.txt")
        fle.touch(exist_ok=True)
        f = open(f"../Renders/{reference_data}_v{net_version}_{exp}_exp_{width}_{height}/data.txt", 'a')
        print("RE =     ", RE01, RE05, RE1, RE15, flush=True)
        print("RESTDS = ", RESTD01, RESTD05, RESTD1, RESTD15, flush=True)
        f.write(f"{angle},{RMSE},{RE01},{RESTD01},{RE05},{RESTD05},{RE1},{RESTD1},{RE15},{RESTD15},{qnet_time},{voxel_time}\n")
        f.close()
        # plt.show()
    comm.Barrier()
 