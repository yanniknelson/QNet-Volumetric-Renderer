import numpy as np
# import matplotlib.pyplot as plt
import openvdb as vdb
from scipy.io import savemat
# import pandas as pd
# #export LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/"


filename="Frame119"

print(f"volumes/{filename}.vdb")

grid = vdb.readAllGridMetadata(f"volumes/{filename}.vdb")[0]
    
grid = vdb.read(f"volumes/{filename}.vdb", grid.name)

print("read data")

(active_lower, active_higher) = grid.evalActiveVoxelBoundingBox()
print(active_lower, active_higher)

size = (int(active_higher[0]-active_lower[0]), int(active_higher[1]-active_lower[1]), int(active_higher[2]-active_lower[2]))

array = np.ndarray(size, dtype=float)

print(size)

grid.copyToArray(array, ijk=grid.evalActiveVoxelBoundingBox()[0])

np.save("volumes/npversions/" + filename + ".npy", array)

data = []

for x in range(size[0]):
    for y in range(size[1]):
        for z in range(size[2]):
            data.append([(x/(size[0]-1))*2 - 1, (y/(size[1]-1))*2 - 1, (z/(size[2]-1))*2 - 1, array[x][y][z]])

mdic = {"a": data, "label": "experiment"}

savemat(f"volumes/{filename}.mat", mdic)

# plt.imshow(array[np.shape(array)[0]//2])
# plt.show()


# from tokenize import Double
# import numpy as np
# # import matplotlib.pyplot as plt
# import openvdb as vdb
# import pandas as pd
# from dataclasses import make_dataclass
# import pickle5 as pickle
# from scipy.io import savemat
# #export LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/"

# min_corner = np.array([float('inf'),float('inf'),float('inf')])
# max_corner = np.array([float('-inf'),float('-inf'),float('-inf')])


# print("working")

# for i in range(1,32):
#     grid = vdb.readAllGridMetadata(f"volumes/Anim/vdbData/Frame{i}.vdb")[0]
    
#     grid = vdb.read(f"volumes/Anim/vdbData/Frame{i}.vdb", grid.name)
    
#     (active_lower, active_higher) = grid.evalActiveVoxelBoundingBox()
#     for i in range(3):
#         if active_lower[i] < min_corner[i]:
#             min_corner[i] = active_lower[i]
#         if active_higher[i] > max_corner[i]:
#             max_corner[i] = active_higher[i]

# size = (31, int(max_corner[0]-min_corner[0]), int(max_corner[1]-min_corner[1]),int(max_corner[2]-min_corner[2]))

# print("newSize ", size)


# array = np.ndarray(size, dtype=float)

# for i in range(1,32):
#     print(i)
#     grid = vdb.readAllGridMetadata(f"volumes/Anim/vdbData/Frame{i}.vdb")[0]
#     grid = vdb.read(f"volumes/Anim/vdbData/Frame{i}.vdb", grid.name)
#     print(min_corner)
#     print(np.shape(array[i-1]))
#     grid.copyToArray(array[i-1], ijk=(3,1,6))
#     print(f"read frame {i}")

# np.save("volumes/Anim/Anim.npy", array)

# print("success")

# print("creating dataframe")

# Point = make_dataclass("Point", [("t", int), ("x", int), ("y", int), ("z", int), ("density", Double)])

# data = []

# # convert to training data
# for t in range(size[0]):
#     for x in range(size[1]):
#         for y in range(size[2]):
#             for z in range(size[3]):
#                 data.append([(x/(size[1]-1))*2 - 1, (y/(size[2]-1))*2 - 1, (z/(size[3]-1))*2 - 1, (t/(size[0]-1))*2 - 1, array[t][x][y][z]])

# print(data[0])
# print(data[-1])

# mdic = {"a": data, "label": "experiment"}

# savemat("volumes/Anim/Anim.mat", mdic)