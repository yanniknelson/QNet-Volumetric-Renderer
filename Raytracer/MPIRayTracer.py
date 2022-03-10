from mpi4py import MPI 
from Qnet import *
from Marcher import Marcher
from Camera import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # get your process ID

width = 100
height = 100

pos = np.array([0, 0, 4])
up = np.array([1,0,0])
lookat = np.array([0,0,0])

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

    recv = np.empty((h, w), dtype=np.float)
    # for i in range(size):
    #     recv.append()

    work = comm.scatter(work, root=0)
    qnet = comm.bcast(qnet, root=0)
    marcher = comm.bcast(marcher, root=0)
    

    c = Camera(height, width, 35, pos, up, lookat)
    vol = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))

    print(work[0])

    for (x, y) in work:
        ray = c.GenerateRay(x,y)
        hit, t0, t1 = vol.intersect(ray)
        if hit:
            intersectionPoint = ray.o + t0* ray.d
            recv[y][x - int(rank*(width/size))] = marcher.trace_scaling(intersectionPoint, ray.d)
            # qnet.Transform(intersectionPoint, ray.d)
            # recv[y][x - int(rank*(width/size))] = (torch.sigmoid((qnet.IntegrateRay(ray, t0, t1) + (t1-t0))/2).item())
        else:
            recv[y][x - int(rank*(width/size))] = -float('inf')

data = comm.gather(recv, root=0)

if rank == 0:
    print(len(data))
    image = []
    for y in range(height):
        image.append([])
    for w in range(size):
        for y in range(height):
            for x in range(int(width/size)):
                image[y].append(data[w][y][x])
    im = plt.imshow(np.array(image))
    plt.colorbar(im)
    plt.show()
