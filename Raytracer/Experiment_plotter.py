import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as pltscale
import matplotlib.ticker as pltticker

exp = "zup-down"


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
    axes[0].set_xlabel("Angle around " + exp[0].upper() + " axis")
    axes[1].set_xlabel("Angle around " + exp[0].upper() + " axis")
    axes[2].set_xlabel("Angle around " + exp[0].upper() + " axis")
    axes[3].set_xlabel("Angle around " + exp[0].upper() + " axis")
    axes[0].set_title("Slack = 0.01")
    axes[1].set_title("Slack = 0.05")
    axes[2].set_title("Slack = 0.1")
    axes[3].set_title("Slack = 0.15")
    axes[0].xaxis.set_ticks(np.arange(0, 361, 45))
    max = 2.2#(np.floor(np.max(lines[2] + lines[3])/0.1)+1)*0.1
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


    fig, axes = plt.subplots(1,1, sharex=True)
    axes.set_xlim([0, 360])
    axes.set_ylim([0, 5])
    axes.plot(lines[0], lines[11], label="Ray Marcher Render Time")
    axes.plot(lines[0], lines[10], label="Q-Net Render Time")
    axes.xaxis.set_ticks(np.arange(0, 361, 45))
    axes.yaxis.set_ticks(np.arange(0, 5.5, 0.5))
    axes.set_ylabel("Render Time (s)")
    axes.set_xlabel("Angle around " + exp[0].upper() + " axis")
    axes.legend()
    plt.tight_layout()
    plt.show()

def plot_gpunt():
    gpudata = np.array([convertToFloat(line.rstrip().split(',')) for line in open(f"../Renders/GPUExperiments/GPU/data.txt")]).T
    nogpudata = np.array([convertToFloat(line.rstrip().split(',')) for line in open(f"../Renders/GPUExperiments/No_GPU/data.txt")]).T
    fig, axes = plt.subplots(1,1, sharex=True)
    axes.set_xlim([0, 640**2])
    axes.set_ylim([0, 640])
    points = gpudata[0] * gpudata[0]
    axes.plot(points, gpudata[1], label="GPU")
    axes.plot(points, nogpudata[1], label="CPU")
    # axes.xaxis.set_ticks(np.logspace(100, 640**2, 7, endpoint=True, base=4))
    axes.yaxis.set_ticks(np.arange(0, 641, 40))
    axes.set_ylabel("Render Time (s)")
    axes.set_xlabel("Number of Pixels")
    axes.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1,1, sharex=True)
    axes.set_xlim([0, 640**2])
    axes.set_ylim([0, 19])
    axes.plot(points, gpudata[1]/nogpudata[1])
    # axes.xaxis.set_ticks(np.logspace(100, 640**2, 7, base=4))
    axes.yaxis.set_ticks(np.arange(0, 19.5, 1))
    axes.set_ylabel("GPU render time / CPU render time")
    axes.set_xlabel("Number of Pixels")
    plt.tight_layout()
    plt.show()



def main(argv):
    if argv[0] == "-e":
        if argv[1] == "-s":
            plot_exp(f"../Renders/static_Blender_cloud_v1_{argv[2]}_exp_{argv[3]}_{argv[3]}/data.txt")
        else:
            plot_exp(f"../Renders/Blender_cloud_v1_{argv[1]}_exp_{argv[2]}_{argv[2]}/data.txt")
    if argv[0] == "-gputnt":
        plot_gpunt()

if __name__ == "__main__":
   main(sys.argv[1:])
