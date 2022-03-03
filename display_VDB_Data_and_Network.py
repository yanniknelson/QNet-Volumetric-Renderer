import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.widgets import Slider
import scipy.io as scio

df = pd.read_pickle("fluid_data_0083_dataframe.pkl")

weights = scio.loadmat("MATLABtest/volume_weights.mat")
pw1 = weights["pw1"]
pb1 = weights["pb1"]
pw2 = weights["pw2"]
pb2 = weights["pb2"]

model = torch.nn.Sequential(torch.nn.Linear(3,500), torch.nn.Sigmoid(), torch.nn.Linear(500,1))

model[0].weight = torch.nn.Parameter(torch.tensor(pw1, dtype=torch.float32))
model[0].bias = torch.nn.Parameter(torch.squeeze(torch.tensor(pb1, dtype=torch.float32)))
model[2].weight = torch.nn.Parameter(torch.tensor(pw2, dtype=torch.float32))
model[2].bias = torch.nn.Parameter(torch.squeeze(torch.tensor(pb2, dtype=torch.float32)))

data = np.empty((63*117, 3))


for i in range(63):
    for j in range(117):
        data[i*117 + j] = [0, (j/58)-1, (i/31)-1]

data = torch.tensor(data, dtype=torch.float32)


res = np.reshape(model(data).detach().numpy(),(63, 117))

res = (res+1)/2

array = np.load("fluid_data_0083_numpy_array.npy")

fig, (ax1, ax2) = plt.subplots(1,2)

plot1 = ax1.imshow(array[58].T, origin="lower")
plot2 = ax2.imshow(res, origin="lower")
# ax1.subplots_adjust( bottom=0.25)

axamp = plt.axes([0.1, 0.1, 0.65, 0.03])
slice_slider = Slider(
    ax=axamp,
    label="slice",
    valmin=0,
    valmax=116,
    valinit=58
)

def update(val):
    plot1.set_data(array[int(val)].T)
    for i in range(63*117):
        data[i][0] = (val/58)-1
    
    res = np.reshape(model(data).detach().numpy(),(63, 117))
    res = (res+1)/2
    print("data min and max: ", np.min(array[int(val)]), np.max(array[int(val)]))
    print("neural min and max: ", np.min(res), np.max(res))
    print()
    plot2.set_data(res)

slice_slider.on_changed(update)


plt.show()