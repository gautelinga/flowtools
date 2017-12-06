import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot some interpolated data")
    parser.add_argument("field_folder_in", type=str, help="Field folder in")
    parser.add_argument("tstep", type=int, help="Timestep")
    parser.add_argument("probes_file_in", type=str, help="Probes file in")
    parser.add_argument("normals_file_in", type=str, help="Normals file in")
    parser.add_argument("point", type=int, help="Point id")
    args = parser.parse_args()

    with h5py.File(args.field_folder_in + "/u_" +
                   str(args.tstep) + ".h5", "r") as h5f:
        u_data = np.array(h5f["u/" + str(args.tstep)])
        p_data = np.array(h5f["p/" + str(args.tstep)])
        grad_u_data = np.array(h5f["grad_u/" + str(args.tstep)])
        sigma_data = np.zeros((len(u_data), 6))
        sigma_data[:, 0] = grad_u_data[:, 0]
        sigma_data[:, 1] = grad_u_data[:, 4]
        sigma_data[:, 2] = grad_u_data[:, 8]
        sigma_data[:, 3] = 0.5*(grad_u_data[:, 1]+grad_u_data[:, 3])
        sigma_data[:, 4] = 0.5*(grad_u_data[:, 2]+grad_u_data[:, 6])
        sigma_data[:, 5] = 0.5*(grad_u_data[:, 5]+grad_u_data[:, 7])

    with h5py.File(args.probes_file_in, "r") as h5f:
        x = np.array(h5f["x"])

    with h5py.File(args.normals_file_in, "r") as h5f:
        n = np.array(h5f["n"])
        x_n = np.array(h5f["x"])
        t_x = np.array(h5f["t_x"])
        t_s = np.array(h5f["t_s"])

    pt_xy = x_n[args.point, :2]
    pt_ids = (x[:, 0] == pt_xy[0])*(x[:, 1] == pt_xy[1])

    z_pt = x[pt_ids, 2]
    u_pt = u_data[pt_ids, :]
    stress_pt = sigma_data[pt_ids, :]
    p_pt = p_data[pt_ids]

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(z_pt, u_pt[:, 0])
    ax.plot(z_pt, u_pt[:, 1])
    ax.plot(z_pt, u_pt[:, 2])

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(z_pt, p_pt)
    ax.plot(z_pt, stress_pt[:, 0])
    ax.plot(z_pt, stress_pt[:, 1])
    ax.plot(z_pt, stress_pt[:, 2])
    #ax.plot(z_pt, stress_pt[:, 3])
    #ax.plot(z_pt, stress_pt[:, 4])
    #ax.plot(z_pt, stress_pt[:, 5])

    x_first = x[0, :2]
    nz = np.sum((x[:, 0] == x_first[0])*(x[:, 1] == x_first[1]))
    Nxy = len(x)/nz

    u_dict = dict()
    for dim in range(3):
        u_dict[dim] = dict()
        for i in range(Nxy):
            ind = nz*i
            xy = (x[ind, 0], x[ind, 1])
            u_dict[dim][xy] = np.mean(u_data[ind:ind+nz, dim])

    x_ = np.unique(x_n[:, 0])[:]
    y_ = np.unique(x_n[:, 1])[:]
    X, Y = np.meshgrid(x_, y_)
    u_mean = dict()
    for dim in range(3):
        u_mean[dim] = np.zeros_like(X)
    for ix, x_val in enumerate(x_):
        for iy, y_val in enumerate(y_):
            for dim in range(3):
                u_mean[dim][ix, iy] = u_dict[dim][(x_val, y_val)]

    for dim in range(3):
        fig = plt.figure()
        plt.imshow(u_mean[dim], cmap='hot', interpolation='nearest')

    
        
    plt.show()


if __name__ == "__main__":
    main()
