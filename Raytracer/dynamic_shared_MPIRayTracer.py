from matplotlib.pyplot import axes
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

total = width * height

pos = np.array([0, 0, 4])
up = np.array([1,0,0])
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

if rank == 0:
    counter[0] = batchsize * size

start = rank * batchsize

qnet_time = 0
voxel_time = 0

comm.Barrier()

with torch.no_grad():

    if rank == 0:
        weights = scio.loadmat("../MATLABtest/volume_weights_v2.mat")

        qnet = Intergrator(weights["pw1"], weights["pb1"], weights["pw2"], weights["pb2"])

        marcher = Marcher(np.array([-1,-1,-1]), np.array([1,1,1]), "../fluid_data_0083_numpy_array.npy")

    else:
        marcher = Marcher()
        qnet = Intergrator()

    qnet = comm.bcast(qnet, root=0)
    marcher = comm.bcast(marcher, root=0)

    c = Camera(height, width, 35, pos, up, lookat)
    vol = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))

    go_again = True

    if rank == 0:
        print("qnet start", flush=True)

    comm.Barrier()

    if rank == 0:
        start_time = MPI.Wtime()

    while go_again:
        for i in range(start, min(start + batchsize, total)):#int((start+batchsize)%(total+1))):
            y = i//width
            x = i%height
            ray = c.GenerateRay(x,y)
            hit, t0, t1 = vol.intersect(ray)
            if hit:
                # image[y][x] = torch.sigmoid((qnet.IntegrateRay(ray, t0, t1) + (t1-t0))/2).item()#qnet.IntegrateRay(ray, t0, t1) + (t1-t0)*0.4
                image[y][x] = max(torch.sigmoid((qnet.IntegrateRay(ray, t0, t1) + (t1-t0))/2).item() - 0.5, 0)*(4*0.9)

        go_again = False

        counter_win.Wait()
        counter_win.Lock_all()
        if counter[0] < total:
            go_again = True
            start = int(counter[0])
            counter[0] = int((start + batchsize) % (total+1))
        counter_win.Unlock_all()

comm.Barrier()

if rank == 0:
    end_time = MPI.Wtime()
    qnet_time = end_time - start_time
    print("Qnet render finished in: ", qnet_time,"\nvoxel start", flush=True)

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
