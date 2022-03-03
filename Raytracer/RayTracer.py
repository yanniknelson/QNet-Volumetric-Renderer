import matplotlib.pyplot as plt
from alive_progress import alive_bar
from Camera import *

image = []


width = 800
height = 800
pos = np.array([-4, -1, 2])
up = np.array([0,0,1])
lookat = np.array([0,0,0])

c = Camera(height, width, 35, pos, up, lookat)
test = Bounds3(np.array([-1,1,1]), np.array([1,-1,-1]))
with alive_bar(width * height) as bar:
    for y in range(height):
        image.append([])
        for x in range(width):
            ray = c.GenerateRay(x,y)
            if test.intersect(ray)[0]:
                image[y].append(1)
            else:
                image[y].append(0)
            bar()

plt.imshow(np.array(image))
plt.show()
