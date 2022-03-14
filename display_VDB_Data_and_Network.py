import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.widgets import Slider
import scipy.io as scio

weights = scio.loadmat("MATLABtest/volume_weights_v3.mat")
pw1 = weights["pw1"]
pb1 = weights["pb1"]
pw2 = weights["pw2"]
pb2 = weights["pb2"]

model = torch.nn.Sequential(torch.nn.Linear(3,500), torch.nn.Sigmoid(), torch.nn.Linear(500,1))

model[0].weight = torch.nn.Parameter(torch.tensor(pw1, dtype=torch.float32))
model[0].bias = torch.nn.Parameter(torch.squeeze(torch.tensor(pb1, dtype=torch.float32)))
model[2].weight = torch.nn.Parameter(torch.tensor(pw2, dtype=torch.float32))
model[2].bias = torch.nn.Parameter(torch.squeeze(torch.tensor(pb2, dtype=torch.float32)))

xslicedata = np.empty((63*117, 3))
yslicedata = np.empty((63*117, 3))


for i in range(63):
    for j in range(117):
        xslicedata[i*117 + j] = [0, (j/58)-1, (i/31)-1]
        yslicedata[i*117 + j] = [(j/58)-1, 0, (i/31)-1]

xslicedata = torch.tensor(xslicedata, dtype=torch.float32)
yslicedata = torch.tensor(yslicedata, dtype=torch.float32)

array = np.load("fluid_data_0083_numpy_array.npy")

mx = np.max(array)
mn = np.min(array)

scl = 2/(mx-mn)


xres = np.reshape(model(xslicedata).detach().numpy(),(63, 117))
yres = np.reshape(model(yslicedata).detach().numpy(),(63, 117))


fig, axs = plt.subplots(2,2)

plot1 = axs[0,0].imshow(array[58].T, origin="lower")
plot2 = axs[0,1].imshow(xres, origin="lower")

plot3 = axs[1,0].imshow(array[:][58].T, origin="lower")
plot4 = axs[1,1].imshow(yres, origin="lower")
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
    plot3.set_data(array[:][int(val)].T)
    for i in range(63*117):
        xslicedata[i][0] = (val/58)-1
        yslicedata[i][1] = (val/58)-1
    
    xres = np.reshape(model(xslicedata).detach().numpy(),(63, 117))
    yres = np.reshape(model(yslicedata).detach().numpy(),(63, 117))
    print("data min and max: ", np.min(array[int(val)]), np.max(array[int(val)]))
    print("neural min and max: ", np.min(xres), np.max(xres))
    print("neural scaled min and max: ", (np.min(xres)+1)/scl, (np.max(xres)+1)/scl)
    print()
    plot2.set_data(xres)
    plot4.set_data(yres)

slice_slider.on_changed(update)


plt.show()