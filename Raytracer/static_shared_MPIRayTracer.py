from mpi4py import MPI 
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

pos = np.array([0, 0, 4])
up = np.array([1,0,0])
lookat = np.array([0,0,0])

qnet_time = 0
voxel_time = 0

start_time = None

dtype = MPI.DOUBLE
np_dtype = dtlib.to_numpy_dtype(MPI.DOUBLE)
image_win_size = ((MPI.DOUBLE.Get_size() * width * height) if (rank == 0) else (0))
image_win = MPI.Win.Allocate_shared(image_win_size, MPI.DOUBLE.Get_size(), comm=comm)
buf, itemsize = image_win.Shared_query(0) 
image = np.ndarray(buffer=buf, dtype='d', shape=(height,width))

#create reference image shared memory
ref_win = MPI.Win.Allocate_shared(image_win_size, dtype.Get_size(), comm=comm)
buf, itemsize = ref_win.Shared_query(0) 
ref = np.ndarray(buffer=buf, dtype='d', shape=(height,width))

#create volume data shared memory
vol_win_size = 0
volume = None
shape = 0
if rank == 0:
    volume = np.load("../fluid_data_0083_numpy_array.npy")
    vol_win_size = dtype.Get_size() * np.size(volume)
    shape = np.shape(volume)

shape = comm.bcast(shape, root=0)

vol_win = MPI.Win.Allocate_shared(vol_win_size, dtype.Get_size(), comm=comm)
buf, itemsize = vol_win.Shared_query(0) 
volume_data = np.ndarray(buffer=buf, dtype='d', shape=shape)

if rank == 0:
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                volume_data[i][j][k] = volume[i][j][k]
    volume = None

marcher = Marcher(np.array([-1,-1,-1]), np.array([1,1,1]), volume_data)

with torch.no_grad():

    work = None
    if rank == 0:
        weights = scio.loadmat("../MATLABtest/volume_weights_v2.mat")
        
        qnet = Intergrator(weights["pw1"], weights["pb1"], weights["pw2"], weights["pb2"])

        work = []
        for w in range(size):
            work.append([])
            for y in range(height):
                xstart = int(w*(width/size))
                xend = int((w+1)*(width/size))
                for x in range(xstart, xend):
                    work[w].append((x,y))
    else:
        qnet = Intergrator()

    work = comm.scatter(work, root=0)
    qnet = comm.bcast(qnet, root=0)

    c = Camera(height, width, 35, pos, up, lookat)
    vol = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))

    if rank == 0:
        print("qnet start", flush=True)

    comm.Barrier()

    if rank == 0:
        start_time = MPI.Wtime()

    for (x, y) in work:
        ray = c.GenerateRay(x,y)
        hit, t0, t1 = vol.intersect(ray)
        if hit:
            image[y][x] = max(torch.sigmoid((qnet.IntegrateRay(ray, t0, t1) + (t1-t0))/2).item() - 0.5, 0)*(4*0.9)

comm.Barrier()

if rank == 0:
    end_time = MPI.Wtime()
    qnet_time = end_time - start_time
    print("Qnet render finished in: ", qnet_time,"\nvoxel start", flush=True)

comm.Barrier()

if rank == 0:
    start_time = MPI.Wtime()

for (x, y) in work:
    ray = c.GenerateRay(x,y)
    hit, t0, t1 = vol.intersect(ray)
    if hit:
        ref[y][x] = marcher.trace_scaling(ray.o + t0* ray.d, ray.d)

comm.Barrier()

if rank == 0:
    end_time = MPI.Wtime()
    voxel_time = end_time - start_time

if rank == 0:
    fig, axes = plt.subplots(1,2)
    print("qnet render time = ", qnet_time, flush=True)
    print("voxel render time = ", voxel_time, flush=True)
    RMSE = np.sqrt(np.mean((ref-image)**2))
    print("RMSE = ", RMSE, flush=True)
    print("RE = ", RMSE/np.mean(ref), flush=True)
    im = axes[0].imshow(np.array(image))
    fig.colorbar(im, ax=axes[0])
    rf = axes[1].imshow(np.array(ref))
    fig.colorbar(rf, ax=axes[1])
    plt.show()
