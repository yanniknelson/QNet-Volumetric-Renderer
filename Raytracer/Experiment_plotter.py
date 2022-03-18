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
    lines = np.array(lines).T
    # print(np.shape(lines))
    fig, axes = plt.subplots(1,3, sharex=True)
    axes[0].set_ylabel(r"Relavite Error $\mathrm{mean}\left(\frac{|\mathrm{measured}  - \mathrm{reference}|}{\mathrm{reference} + \mathrm{slack}}\right)$")
    axes[0].set_xlabel("Angle around Y axis")
    axes[1].set_xlabel("Angle around Y axis")
    axes[2].set_xlabel("Angle around Y axis")
    axes[0].set_title("Slack = 0.05")
    axes[1].set_title("Slack = 0.1")
    axes[2].set_title("Slack = 0.15")
    axes[0].xaxis.set_ticks(np.arange(0, 361, 60))
    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[2].set_ylim([0, 1])
    axes[0].yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    axes[1].yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    axes[2].yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    axes[0].plot(lines[0], lines[2], label="Slack = 0.05")
    axes[0].fill_between(x=lines[0], y1=np.maximum(lines[2] - lines[3],np.zeros(np.shape(lines[0]))), y2 = lines[2]+lines[3], alpha=0.5)
    axes[1].plot(lines[0], lines[4], label="Slack = 0.1", c='orange')
    axes[1].fill_between(lines[0], y1=np.maximum(lines[4] - lines[5],np.zeros(np.shape(lines[0]))), y2 = lines[4]+lines[5], alpha=0.5, color='orange')
    axes[2].plot(lines[0], lines[6], label="Slack = 0.15", c='green')
    axes[2].fill_between(lines[0], y1=np.maximum(lines[6] - lines[7],np.zeros(np.shape(lines[0]))), y2 = lines[6]+lines[7], alpha=0.3, color='green')
    plt.tight_layout()
    plt.show()


    plt.plot(lines[0], lines[9], label="Q-Net Render Time")
    plt.plot(lines[0], lines[8], label="Ray Marcher Render Time")
    plt.xticks(np.arange(0, 361, 60))
    plt.ylabel("Render Time")
    plt.xlabel("Angle around Y axis")
    plt.legend()
    plt.show()

plot_exp("../Renders/Blender_cloud_v1_y_exp_400_400/data.txt")
