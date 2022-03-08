from alive_progress import alive_bar
from torch import true_divide
from Qnet import *
from Camera import *

image = []


width = 200
height = 200
pos = np.array([0, 4, 2])
up = np.array([0,0,1])
lookat = np.array([0,0,0])

with torch.no_grad():
    weights = scio.loadmat("../MATLABtest/volume_weights.mat")
    pw1 = torch.tensor(weights["pw1"], dtype=type, device=device)
    pb1 = torch.squeeze(torch.tensor(weights["pb1"], dtype=type, device=device))
    pw2 = torch.tensor(weights["pw2"], dtype=type, device=device)
    pb2 = torch.squeeze(torch.tensor(weights["pb2"], dtype=type, device=device))

    qnet = Intergrator(pw1, pb1, pw2, pb2)

    c = Camera(height, width, 35, pos, up, lookat)
    vol = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))
    with alive_bar(width * height) as bar:
        for y in range(height):
            image.append([])
            for x in range(width):
                ray = c.GenerateRay(x,y)
                hit, t0, t1 = vol.intersect(ray)
                if hit:
                    qnet.Transform(ray.o + t0 * ray.d, ray.d)
                    image[y].append(torch.sigmoid((qnet.apply(0, t1-t0) + (t1-t0))/2).item())
                else:
                    image[y].append(-float('inf'))
                bar()

im = plt.imshow(np.array(image))
plt.colorbar(im)
plt.show()
