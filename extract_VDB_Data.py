import numpy as np
import matplotlib.pyplot as plt
import openvdb as vdb
import matplotlib.pylab as plt
#export LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/"

filename="Blender_cloud"
grid = vdb.readAllGridMetadata("volumes/" + filename + ".vdb")[0]

print(grid)
print(grid.name)

grid = vdb.read(filename + ".vdb", grid.name)

print("read data")

(active_lower, active_higher) = grid.evalActiveVoxelBoundingBox()
print(active_lower, active_higher)

size = (active_higher[0]-active_lower[0], active_higher[1]-active_lower[1],active_higher[2]-active_lower[2])

array = np.ndarray(size, float)

grid.copyToArray(array, ijk=active_lower)

np.save("volumes/npversions/" + filename + ".npy", array)

plt.imshow(array[np.shape(array)[0]//2])
plt.show()

