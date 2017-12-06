import h5py
from mpi4py import MPI
import numpy as np
import argparse
import os


def make_hist(node_data, node_vol):
    hist, bin_edges = np.histogram(node_data, weights=node_vol,
                                   density=True, bins=256)
    bins = 0.5*(bin_edges[:-1]+bin_edges[1:])
    return hist, bins


def save_hist(node_data, node_vol, filename):
    hist, bins = make_hist(node_data, node_vol)
    np.savetxt(filename, np.vstack((bins, hist)).T)


def main():
    u_folder = "VisualisationVector"

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description="Compute flow properties.")
    parser.add_argument('filename_w', type=str,
                        help='mesh weights filename')
    parser.add_argument('folder', type=str,
                        help='velocity and pressure folder')
    parser.add_argument('-dset_start', type=int, default=0,
                        help='index of dataset to start with')
    parser.add_argument('-l', '--list_dsets', action="store_true",
                        help='list datasets')
    parser.add_argument('-hist', action="store_true",
                        help='create histograms')
    parser.add_argument('-avg', action="store_true",
                        help='compute averages')
    parser.add_argument('-flow_dir', type=int, default=0,
                        help="flow direction")
    args = parser.parse_args()
    filename_w = args.filename_w
    filename_u = args.folder + "/u_from_tstep_0.h5"
    filename_p = args.folder + "/p_from_tstep_0.h5"
    filename_out_time_avg = args.folder + "/time_avg.dat"

    nu = 9.e-6

    if not os.path.isfile(filename_w):
        exit("Couldn't find weights file:" + filename_w)
    if not os.path.isfile(filename_u):
        exit("Couldn't find velocity file:" + filename_u)
    if not os.path.isfile(filename_p):
        exit("Couldn't find pressure file:" + filename_p)

    h5f_w = h5py.File(filename_w, 'r', driver='mpio', comm=comm)
    h5f_u = h5py.File(filename_u, 'r', driver='mpio', comm=comm)
    h5f_p = h5py.File(filename_p, 'r', driver='mpio', comm=comm)

    comm.Barrier()
    if "node_vol" in h5f_w["data"]:
        if rank == 0:
            print "Found node_vol"
        node_vol = np.array(h5f_w["data/node_vol"])
        vol = np.sum(node_vol)
        if rank == 0:
            print "Total volume =", vol

        u_avg = np.zeros(3)
        u2_avg = np.zeros(3)
        u = np.zeros((len(node_vol), 3))
        p = np.zeros((len(node_vol), 1))
        dsets = [int(key) for key in h5f_u[u_folder].keys()]
        n_dsets = len(dsets)
        if args.list_dsets and rank == 0:
            print dsets
    
        data_proc = np.zeros((n_dsets, 9))
        print n_dsets, "datasets"

        for timestep in xrange(rank+args.dset_start,
                               n_dsets, size):
            u[:, :] = h5f_u[u_folder][str(timestep)]
            u_abs = np.sqrt(u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2)
            p[:, :] = h5f_p[u_folder][str(timestep)]
            if args.avg:
                for dim in xrange(3):
                    u_avg[dim] = np.sum(node_vol*u[:, dim])/vol
                    u2_avg[dim] = np.sum(node_vol*u[:, dim]**2)/vol
                Re = 1.*u_avg[args.flow_dir]/nu
                print timestep, u_avg[0], u_avg[1], u_avg[2], Re
                data_proc[timestep, 0] = timestep
                data_proc[timestep, 1:4] = u_avg[:]
                data_proc[timestep, 4] = Re
                data_proc[timestep, 5:8] = u2_avg[:]
                data_proc[timestep, 8] = np.sqrt(np.max(u[:, 0]**2+u[:, 1]**2))
            if args.hist:
                for dim in xrange(3):
                    save_hist(u[:, dim], node_vol,
                              args.folder + "/u_" + str(dim) +
                              "_dset" + str(timestep) + ".hist")
                save_hist(u_abs, node_vol,
                          args.folder + "/u_abs_dset" +
                          str(timestep) + ".hist")
                save_hist(np.log10(u_abs[u_abs > 1e-4]),
                          node_vol[u_abs > 1e-4],
                          args.folder + "/log_u_abs_dset" +
                          str(timestep) + ".hist")
                save_hist(p[:, 0], node_vol,
                          args.folder + "/p_dset" +
                          str(timestep) + ".hist")
                p_abs = np.abs(p[:, 0])
                save_hist(np.log10(p_abs[p_abs > 1e-8]),
                          node_vol[p_abs > 1e-8],
                          args.folder + "/log_p_abs_dset" +
                          str(timestep) + ".hist")
   
                for i, j in [[0, 1], [0, 2], [1, 2]]:
                    hist, x_edges, y_edges = np.histogram2d(
                        u[:, i], u[:, j], bins=60, normed=True,
                        weights=node_vol)
                    x_bins = 0.5*(x_edges[:-1]+x_edges[1:])
                    y_bins = 0.5*(y_edges[:-1]+y_edges[1:])

                    hist2dfile = open(args.folder + "/u_" + str(i) +
                                      "-" + str(j) + "_dset" +
                                      str(timestep) + ".hist2d", "w")
                    for ix, x in enumerate(x_bins):
                        for iy, y in enumerate(y_bins):
                            hist2dfile.write(
                                str(x) + " " + str(y) + " " +
                                str(hist[ix, iy]) + "\n")
                        hist2dfile.write("\n")
                    hist2dfile.close()

    comm.Barrier()

    if rank == 0:
        data = np.zeros_like(data_proc)
    else:
        data = None

    comm.Reduce(
        [data_proc, MPI.DOUBLE],
        [data, MPI.DOUBLE],
        op=MPI.SUM, root=0)

    if rank == 0 and args.avg:
        np.savetxt(filename_out_time_avg, data)

    h5f_w.close()
    h5f_u.close()
    h5f_p.close()


if __name__ == "__main__":
    main()
