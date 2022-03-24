import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import scipy.io as scio

weights = scio.loadmat("MATLABtest/anim_weights_v1.mat")
pw1 = weights["pw1"]
pb1 = weights["pb1"]
pw2 = weights["pw2"]
pb2 = weights["pb2"]

scl = weights["yrange"]

model = torch.nn.Sequential(torch.nn.Linear(3,500), torch.nn.Sigmoid(), torch.nn.Linear(500,1))

model[0].weight = torch.nn.Parameter(torch.tensor(pw1, dtype=torch.float32))
model[0].bias = torch.nn.Parameter(torch.squeeze(torch.tensor(pb1, dtype=torch.float32)))
model[2].weight = torch.nn.Parameter(torch.tensor(pw2, dtype=torch.float32))
model[2].bias = torch.nn.Parameter(torch.squeeze(torch.tensor(pb2, dtype=torch.float32)))

yslicedata = np.empty((54*102, 4))
xslicedata = np.empty((60*102, 4))


# full_data = np.empty((117, 117, 63, 3))
# for x in range(117):
#     for y in range(117):
#         for z in range(63):
#             full_data[x, y, z] = [((x/116)*2)-1,((y/116)*2)-1,((z/62)*2)-1]

# full_data = full_data.reshape((117*117*63, 3))

# data = scio.loadmat("MATLABtest/volume_data.mat")
# data = data['a']
# print(np.shape(data))

# points =  data[:,:3]
# print(np.shape(points))
# points = torch.tensor(points, dtype=torch.float32)


# data = data[:,3].reshape((862407,1))

# print("points ", np.shape(points.numpy()), " data", np.shape(data))

# res = model(points).detach().numpy()

array = np.load("volumes/npversions/Blender_cloud.npy")

# mx = np.max(array)
# mn = np.min(array)

# scl = 2/(mx-mn)

# res = (res + 1)/scl

# print(np.mean((res - data)**2))


for x in range(60):
    for z in range(102):
        xslicedata[z*60 + x] = [0, (x/59)*2-1, (z/101)*2-1,0]

for x in range(54):
    for z in range(102):
        #xslicedata[x*54 + z] = [0, (x*2/53)-1, (z*2/101)-1,0]
        yslicedata[z*54 + x] = [(x/53)*2-1, 0, (z/101)*2-1,0]

xslicedata = torch.tensor(xslicedata, dtype=torch.float32)
yslicedata = torch.tensor(yslicedata, dtype=torch.float32)


xres = np.reshape(model(xslicedata).detach().numpy(),(102, 60))
yres = np.reshape(model(yslicedata).detach().numpy(),(102, 54))


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
    valmin=-1,
    valmax=1,
    valinit=0
)

axamp = plt.axes([0.1, 0.8, 0.65, 0.03])
time_slider = Slider(
    ax=axamp,
    label="slice",
    valmin=-1,
    valmax=1,
    valinit=0
)

def update(val):
    plot1.set_data(array[int(val)].T)
    plot3.set_data(array[:][int(val)].T)
    for i in range(54*102):
        yslicedata[i][3] = val

    for i in range(60*102):
        xslicedata[i][3] = val
    
    xres = np.reshape(model(xslicedata).detach().numpy(),(102, 60))
    yres = np.reshape(model(yslicedata).detach().numpy(),(102, 54))
    print("data min and max: ", np.min(array[int(val)]), np.max(array[int(val)]))
    print("neural min and max: ", np.min(yres), np.max(yres))
    print("neural scaled min and max: ", (np.min(yres)+1)/scl, (np.max(yres)+1)/scl)
    print()
    plot2.set_data(xres)
    plot4.set_data(yres)

time_slider.on_changed(update)

def update2(val):
    plot1.set_data(array[int(val)].T)
    plot3.set_data(array[:][int(val)].T)
    for i in range(54*102):
        yslicedata[i][1] = val

    for i in range(60*102):
        xslicedata[i][0] = val
    
    xres = np.reshape(model(xslicedata).detach().numpy(),(102, 60))
    yres = np.reshape(model(yslicedata).detach().numpy(),(102, 54))
    print("data min and max: ", np.min(array[int(val)]), np.max(array[int(val)]))
    print("neural min and max: ", np.min(yres), np.max(yres))
    print("neural scaled min and max: ", (np.min(yres)+1)/scl, (np.max(yres)+1)/scl)
    print()
    plot2.set_data(xres)
    plot4.set_data(yres)

slice_slider.on_changed(update2)

plt.show()