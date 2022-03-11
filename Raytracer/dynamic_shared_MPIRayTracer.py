from mpi4py import MPI 
from mpi4py.util import dtlib
from Qnet import *
from Marcher import Marcher
from Camera import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # get your process ID

np.seterr(divide='ignore')

width = 100
height = 100

total = width * height

pos = np.array([4, 4, 2])
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

comm.Barrier()

with torch.no_grad():

    if rank == 0:
        weights = scio.loadmat("../MATLABtest/volume_weights_v1.mat")
        pw1 = torch.tensor(weights["pw1"], dtype=type, device=device)
        pb1 = torch.squeeze(torch.tensor(weights["pb1"], dtype=type, device=device))
        pw2 = torch.tensor(weights["pw2"], dtype=type, device=device)
        pb2 = torch.squeeze(torch.tensor(weights["pb2"], dtype=type, device=device))

        qnet = Intergrator(pw1, pb1, pw2, pb2)

        marcher = Marcher(np.array([-1,-1,-1]), np.array([1,1,1]), "../fluid_data_0083_numpy_array.npy")

    else:
        marcher = Marcher()
        qnet = Intergrator()

    h = int(height)
    w = int(width/size)

    qnet = comm.bcast(qnet, root=0)
    marcher = comm.bcast(marcher, root=0)

    comm.Barrier()

    if rank == 0:
        start_time = MPI.Wtime()

    c = Camera(height, width, 35, pos, up, lookat)
    vol = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))

    go_again = True

    while go_again:
        for i in range(start, int((start+batchsize)%(total+1))):
            y = i//width
            x = i%height
            ray = c.GenerateRay(x,y)
            hit, t0, t1 = vol.intersect(ray)
            if hit:
                intersectionPoint = ray.o + t0* ray.d
                # image[y][x] = marcher.trace_scaling(intersectionPoint, ray.d)
                image[y][x] = torch.sigmoid((qnet.IntegrateRay(ray, t0, t1) + (t1-t0))/2).item()
            else:
                image[y][x] = -float('inf')

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

if rank == 0:
    print(end_time-start_time,flush=True)
    im = plt.imshow(np.array(image))
    plt.colorbar(im)
    plt.show()