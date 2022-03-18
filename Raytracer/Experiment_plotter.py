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
    fig, axes = plt.subplots(1,4, sharex=True)
    fig.set_figwidth(15)
    axes[0].set_ylabel(r"Relavite Error $\mathrm{mean}\left(\frac{|\mathrm{measured}  - \mathrm{reference}|}{\mathrm{reference} + \mathrm{slack}}\right)$")
    axes[0].set_xlabel("Angle around X axis")
    axes[1].set_xlabel("Angle around X axis")
    axes[2].set_xlabel("Angle around X axis")
    axes[3].set_xlabel("Angle around X axis")
    axes[0].set_title("Slack = 0.01")
    axes[1].set_title("Slack = 0.05")
    axes[2].set_title("Slack = 0.1")
    axes[3].set_title("Slack = 0.15")
    axes[0].xaxis.set_ticks(np.arange(0, 361, 60))
    max = (np.floor(np.max(lines[2] + lines[3])/0.1)+1)*0.1
    axes[0].set_ylim([0, max])
    axes[1].set_ylim([0, max])
    axes[2].set_ylim([0, max])
    axes[3].set_ylim([0, max])
    axes[0].set_xlim([0, 360])
    axes[1].set_xlim([0, 360])
    axes[2].set_xlim([0, 360])
    axes[3].set_xlim([0, 360])
    axes[0].yaxis.set_ticks(np.arange(0, max+0.1, 0.1))
    axes[1].yaxis.set_ticks(np.arange(0, max+0.1, 0.1))
    axes[2].yaxis.set_ticks(np.arange(0, max+0.1, 0.1))
    axes[3].yaxis.set_ticks(np.arange(0, max+0.1, 0.1))
    axes[0].plot(lines[0], lines[2], label="Slack = 0.01")
    axes[0].fill_between(x=lines[0], y1=np.maximum(lines[2] - lines[3],np.zeros(np.shape(lines[0]))), y2 = lines[2]+lines[3], alpha=0.5)
    axes[1].plot(lines[0], lines[4], label="Slack = 0.05", c='tab:orange')
    axes[1].fill_between(lines[0], y1=np.maximum(lines[4] - lines[5],np.zeros(np.shape(lines[0]))), y2 = lines[4]+lines[5], alpha=0.5, color='tab:orange')
    axes[2].plot(lines[0], lines[6], label="Slack = 0.1", c='green')
    axes[2].fill_between(lines[0], y1=np.maximum(lines[6] - lines[7],np.zeros(np.shape(lines[0]))), y2 = lines[6]+lines[7], alpha=0.5, color='green')
    axes[3].plot(lines[0], lines[8], label="Slack = 0.15", c='red')
    axes[3].fill_between(lines[0], y1=np.maximum(lines[8] - lines[9],np.zeros(np.shape(lines[0]))), y2 = lines[8]+lines[9], alpha=0.5, color='red')
    plt.tight_layout()
    plt.show()


    plt.plot(lines[0], lines[11], label="Q-Net Render Time")
    plt.plot(lines[0], lines[10], label="Ray Marcher Render Time")
    plt.xticks(np.arange(0, 361, 60))
    plt.ylabel("Render Time")
    plt.xlabel("Angle around Y axis")
    plt.legend()
    plt.show()

plot_exp("../Renders/Blender_cloud_v1_x_exp_400_400/data.txt")
