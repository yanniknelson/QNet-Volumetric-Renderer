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

pos = np.array([4, 0, 0])
up = np.array([0,0,1])
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

with torch.no_grad():

    work = None
    if rank == 0:
        weights = scio.loadmat("../MATLABtest/volume_weights_v1.mat")
        pw1 = torch.tensor(weights["pw1"], dtype=type, device=device)
        pb1 = torch.squeeze(torch.tensor(weights["pb1"], dtype=type, device=device))
        pw2 = torch.tensor(weights["pw2"], dtype=type, device=device)
        pb2 = torch.squeeze(torch.tensor(weights["pb2"], dtype=type, device=device))

        qnet = Intergrator(pw1, pb1, pw2, pb2)

        marcher = Marcher(np.array([-1,-1,-1]), np.array([1,1,1]), "../fluid_data_0083_numpy_array.npy")

        work = []
        for w in range(size):
            work.append([])
            for y in range(height):
                xstart = int(w*(width/size))
                xend = int((w+1)*(width/size))
                for x in range(xstart, xend):
                    work[w].append((x,y))
    else:
        marcher = Marcher()
        qnet = Intergrator()

    h = int(height)
    w = int(width/size)

    work = comm.scatter(work, root=0)
    qnet = comm.bcast(qnet, root=0)
    marcher = comm.bcast(marcher, root=0)

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
            image[y][x] = torch.sigmoid((qnet.IntegrateRay(ray, t0, t1) + (t1-t0))/2).item()

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
    print("qnet render time = ", qnet_time, flush=True)
    print("voxel render time = ", voxel_time, flush=True)
    im = plt.imshow(np.array(image))
    plt.colorbar(im)
    plt.show()
    im = plt.imshow(np.array(ref))
    plt.colorbar(im)
    plt.show()
