from tokenize import Double
import numpy as np
# import matplotlib.pyplot as plt
import openvdb as vdb
import pandas as pd
from dataclasses import make_dataclass
import pickle5 as pickle
import seaborn as sns
import matplotlib.pylab as plt
#export LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/"

filename="fluid_data_0083"
# filename = "bunny_cloud"
grid = vdb.readAllGridMetadata(filename + ".vdb")[0]

print(grid)
print(grid.name)

grid = vdb.read(filename + ".vdb", grid.name)

print("read data")

(active_lower, active_higher) = grid.evalActiveVoxelBoundingBox()
print(active_lower, active_higher)
accessor = grid.getAccessor()

size = (active_higher[0]-active_lower[0], active_higher[1]-active_lower[1],active_higher[2]-active_lower[2])

array = np.ndarray(size, float)

grid.copyToArray(array, ijk=active_lower)

np.save(filename + "_numpy_array.npy", array)

print(filename)

print(np.sum(array))


uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data, linewidth=0.5)
plt.show()


print("creating dataframe")

Point = make_dataclass("Point", [("x", int), ("y", int), ("z", int), ("density", Double)])

data = []

# convert to training data
for x in range(size[0]):
    for y in range(size[1]):
        for z in range(size[2]):
            data.append(Point(x, y, z, array[x][y][z]))

df = pd.DataFrame(data)

print(df)

df.to_pickle(filename + "_dataframe.pkl")
