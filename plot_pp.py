import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot something")
    parser.add_argument("folder", type=str, help="Folder")
    args = parser.parse_args()

    u_data = np.loadtxt(os.path.join(args.folder, "u_avg_t.dat"))
    du2_data = np.loadtxt(os.path.join(args.folder, "du2_t.dat"))

    t = u_data[:, 0]
    Re = u_data[:, 1]/9e-6
    Re_q = np.sqrt(du2_data[:, 1])/9e-6

    fig, ax1 = plt.subplots()
    color = "tab:red"

    ax1.set_xlabel("Time")
    ax1.set_ylabel("$Re_t$", color=color)
    ax1.plot(t, Re, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("$Re'_t$")
    ax2.plot(t, Re_q, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()


    fig2, ax22 = plt.subplots()
    ax22.set_xlabel("$Re_t$")
    ax22.set_ylabel("$Re'_t$")
    plt.plot(Re, Re_q)
    
    plt.show()
    
                        
