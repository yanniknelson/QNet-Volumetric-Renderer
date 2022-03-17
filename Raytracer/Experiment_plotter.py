import numpy as np
import matplotlib.pyplot as plt


def convertToFloat(list):
    ret = []
    for i in list:
        ret.append(float(i))
    return ret

def plot_exp(filename):
    f = open(filename, 'r')
    lines = [convertToFloat(line.rstrip().split(',')) for line in f]
    f.close()
    lines = np.array(lines)
    # print(np.shape(lines))
    plt.plot(lines.T[0], lines.T[1])
    # plt.plot(lines.T[0], lines.T[2])
    plt.show()

plot_exp("../Renders/Blender_cloud_v1_z_exp_400_400/data.txt")
