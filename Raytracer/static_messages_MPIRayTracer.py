from mpi4py import MPI 
from Qnet import *
from Marcher import Marcher
from Camera import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # get your process ID

np.seterr(divide='ignore')

width = 100
height = 100

qnet_time = 0
voxel_time = 0

pos = np.array([0, 0, 4])
up = np.array([1,0,0])
lookat = np.array([0,0,0])

start_time = None

with torch.no_grad():

    workdis = None
    if rank == 0:
        weights = scio.loadmat("../MATLABtest/volume_weights_v2.mat")

        qnet = Intergrator(weights["pw1"], weights["pb1"], weights["pw2"], weights["pb2"])

        marcher = Marcher(np.array([-1,-1,-1]), np.array([1,1,1]), "../fluid_data_0083_numpy_array.npy")

        workdis = []
        for w in range(size):
            workdis.append([])
            for y in range(height):
                xstart = int(w*(width/size))
                xend = int((w+1)*(width/size))
                for x in range(xstart, xend):
                    workdis[w].append((x,y))
    else:
        marcher = Marcher()
        qnet = Intergrator()

    comm.barrier()

    work = comm.scatter(workdis, root=0)
    qnet = comm.bcast(qnet, root=0)
    marcher = comm.bcast(marcher, root=0)

    h = int(height)
    w = int(np.shape(work)[0]/height)

    image = np.empty((h, w), dtype=np.float)
    ref = np.empty((h, w), dtype=np.float)

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
            intersectionPoint = ray.o + t0* ray.d
            image[y][x - int(rank*(width/size))] = max(torch.sigmoid((qnet.IntegrateRay(ray, t0, t1) + (t1-t0))/2).item() - 0.5, 0)*(4*0.9)

img = comm.gather(image, root=0)

comm.Barrier()

if rank == 0:
    end_time = MPI.Wtime()
    qnet_time = end_time - start_time
    print("Qnet render finished in: ", qnet_time, "\nvoxel start", flush=True)

comm.Barrier()

if rank == 0:
    start_time = MPI.Wtime()

for (x, y) in work:
    ray = c.GenerateRay(x,y)
    hit, t0, t1 = vol.intersect(ray)
    if hit:
        intersectionPoint = ray.o + t0* ray.d
        ref[y][x - int(rank*(width/size))] = marcher.trace_scaling(intersectionPoint, ray.d)

reference = comm.gather(ref, root=0)

comm.Barrier()

if rank == 0:
    end_time = MPI.Wtime()
    voxel_time = end_time - start_time

if rank == 0:
    image = []
    ref = []
    for y in range(height):
        image.append([])
        ref.append([])
    for w in range(size):
        for y in range(height):
            for x in range(int(np.shape(workdis[w])[0]/height)):
                image[y].append(img[w][y][x])
                ref[y].append(reference[w][y][x])
    image = np.array(image)
    ref = np.array(ref)
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
