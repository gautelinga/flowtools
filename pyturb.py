import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from mpi4py import MPI
import copy


def get_settings():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        args = get_settings_serial()
    else:
        args = None
    args = comm.bcast(args, root=0)
    return args


def get_settings_serial():
    parser = argparse.ArgumentParser(
        description="Plotting flow in a pipe or duct.")
    parser.add_argument("x_file", type=str,
                        help="HDF file for position of probes.")
    parser.add_argument("u_folder", type=str, help="Folder containing u.")
    parser.add_argument("out_folder", type=str, help="Output folder.")
    parser.add_argument("--u_id_max", type=int, help="u field max id.",
                        default=None)
    parser.add_argument("-a", "--append", action="store_true",
                        help="append to file")
    args = parser.parse_args()
    return args


def main():
    args = get_settings()

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    nu = 9e-6
    dx = 2
    dt = 150 * 0.2

    outfilename = os.path.join(args.out_folder, "output.h5")
    if args.append:
        assert os.path.exists(outfilename)

    with h5py.File(args.x_file, "r") as h5f:
        pos = np.array(h5f["x"])

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    dim_x = dict()

    for i in xrange(3):
        dim_x[i] = np.count_nonzero(
            np.all([pos[:, (i+1) % 3] == 0.,
                    pos[:, (i+2) % 3] == 0.], axis=0))

    if mpi_rank == 0:
        print "Dimensions:", dim_x
    dims = (dim_x[0], dim_x[1], dim_x[2])

    x = dict()
    u = dict()
    du = dict()
    u_mean = dict()
    coord = dict()

    for i in xrange(3):
        x[i] = pos[:, i].reshape(dims)
        coord[i] = x[i].mean(((i+1) % 3, (i+2) % 3))
        u[i] = np.zeros_like(x[i])
        du[i] = np.zeros_like(x[i])

    _, _, filenames = os.walk(args.u_folder).next()
    u_ids = []
    for filename in filenames:
        print filename
        file_id = int(filename.split("_")[1].split(".")[0])
        u_ids.append(file_id)

    u_ids.sort()

    if args.append:
        with h5py.File(outfilename, "r") as h5fapp:
            num_series_already = len(h5fapp["uz_centerline"])
            uz_centerline_already = copy.copy(
                np.array(h5fapp["uz_centerline"]))
            q_already = copy.copy(np.array(h5fapp["q"]))
            f_already = copy.copy(np.array(h5fapp["f"]))
            u_mean_already = copy.copy(np.array(h5fapp["u_mean"]))

    num_series = len(u_ids)
    series_proc = range(mpi_rank, num_series, mpi_size)
    num_series_proc = len(series_proc)

    uz_centerline_series = np.zeros((num_series_proc, dims[2]-1))
    q_series = np.zeros((num_series_proc, dims[2]-1))
    f_series = np.zeros((num_series_proc, dims[2]-1))
    u_mean_series = np.zeros((num_series_proc, 3))

    for row_id_proc, row_id in enumerate(series_proc):
        u_id = u_ids[row_id]
        u_id_str = str(u_id)
        if args.append and row_id < num_series_already:
            print "Copying u_" + u_id_str
            uz_centerline_series[row_id_proc, :] = \
                uz_centerline_already[row_id, :]
            q_series[row_id_proc, :] = q_already[row_id, :]
            f_series[row_id_proc, :] = f_already[row_id, :]
            u_mean_series[row_id_proc, :] = u_mean_already[row_id, :]
        else:
            with h5py.File(
                    args.u_folder + "/u_" + u_id_str + ".h5", "r") as h5f:
                try:
                    vel = np.asarray(h5f["u/" + u_id_str])
                except:
                    exit("Couldn't find dataset u/" + u_id_str)
            print "Imported u_" + u_id_str

            for i in xrange(3):
                u[i][:, :] = vel[:, i].reshape(dims)
                u_mean[i] = np.mean(u[i][:-1, :-1, :-1])
                du[i][:, :] = u[i][:, :] - u_mean[i]

            # uz_mean_xy = np.mean(u[2][:, :, :], axis=2)
            # duz_xy = u[2][:, :, dims[2]/2] - uz_mean_xy

            plt.imsave(os.path.join(
                args.out_folder, "uz_" + u_id_str + ".png"),
                       u[2][dims[0]/2, :, :],
                       origin='lower',
                       cmap=plt.get_cmap('viridis'))

            uz_centerline = np.mean(u[2][dims[0]/2-dx:dims[0]/2+dx+1,
                                         dims[1]/2-dx:dims[1]/2+dx+1, :-1],
                                    axis=(0, 1))

            q = np.sqrt(np.mean(
                du[0][:-1, :-1, :-1]**2 + du[1][:-1, :-1, :-1]**2,
                axis=(0, 1)))/u_mean[2]

            ffric_l = np.mean(
                (u[2][1, :-1, :-1]-u[2][0, :-1, :-1]) /
                (coord[0][1]-coord[0][0]),
                axis=0)
            ffric_r = np.mean(
                (u[2][-2, :-1, :-1]-u[2][-1, :-1, :-1]) /
                (coord[0][-1]-coord[0][-2]),
                axis=0)
            ffric_b = np.mean(
                (u[2][:-1, 1, :-1]-u[2][:-1, 0, :-1]) /
                (coord[1][1]-coord[1][0]),
                axis=0)
            ffric_t = np.mean(
                (u[2][:-1, -2, :-1]-u[2][:-1, -1, :-1]) /
                (coord[1][-1]-coord[1][-2]),
                axis=0)
            ffric = (ffric_l + ffric_r + ffric_t)/3.

            uz_centerline_series[row_id_proc, :] = uz_centerline[:]
            q_series[row_id_proc, :] = q[:]
            f_series[row_id_proc, :] = ffric[:]
            u_mean_series[row_id_proc, :] = [u_mean[0], u_mean[1], u_mean[2]]

            # plt.figure() plt.plot(coord[2], q)
            # plt.savefig(os.path.join(args.out_folder, "q_" +
            # u_id_str + ".png")

            # plt.figure()
            # plt.plot(coord[2], u_centerline)
            # plt.savefig(os.path.join(args.out_folder, "uz_centerline_" +
            # u_id_str + ".png"))

            # plt.plot(coord[2],
            #          np.mean(u_centerline) * np.ones_like(u_centerline))
            # plt.plot(coord[2], duz)

            # plt.figure()
            # plt.plot(coord[2], duz_l)
            # plt.plot(coord[2], duz_r)
            # plt.plot(coord[2], duz_t)
            # plt.plot(coord[2], duz_b)
            # plt.plot(coord[2], duz)

    # plt.show()

    h5f = h5py.File(
        outfilename,
        "w", driver="mpio", comm=comm)

    dset_q = h5f.create_dataset("q", (num_series, dims[2]-1))
    dset_f = h5f.create_dataset("f", (num_series, dims[2]-1))
    dset_uz_centerline = h5f.create_dataset(
        "uz_centerline", (num_series, dims[2]-1))
    dset_u_mean = h5f.create_dataset("u_mean", (num_series, 3))

    if num_series_proc > 0:
        rows_proc = xrange(mpi_rank, num_series, mpi_size)
        dset_q[rows_proc, :] = q_series[:, :]
        dset_f[rows_proc, :] = f_series[:, :]
        dset_uz_centerline[rows_proc, :] = uz_centerline_series[:, :]
        dset_u_mean[rows_proc, :] = u_mean_series[:, :]

    h5f.close()

    # np.savetxt(os.path.join(args.out_folder, "u_mean.dat"),
    #            u_mean_series)
    # np.savetxt(os.path.join(args.out_folder, "uz_centerline.dat"),
    #            uz_centerline_series)
    # np.savetxt(os.path.join(args.out_folder, "q.dat"), q_series)
    # np.savetxt(os.path.join(args.out_folder, "f.dat"), f_series)

if __name__ == "__main__":
    main()
