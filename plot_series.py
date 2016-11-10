import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import h5py


def main():
    parser = argparse.ArgumentParser(
        description="Plot timeseries.")
    parser.add_argument("file_in", type=str,
                        help="HDF5 file in.")
    parser.add_argument("timeseries_name", type=str,
                        help="Timeseries name.")
    parser.add_argument("-dir", type=int,
                        help="flow direction axis",
                        default=2)
    parser.add_argument("-dt", type=float,
                        help="Timestep",
                        default=0.2*150)
    parser.add_argument("-nu", type=float,
                        help="Viscosity",
                        default=9e-6)
    parser.add_argument("-L", type=float,
                        help="Length",
                        default=40.0)
    parser.add_argument("-max", type=int,
                        help="max id",
                        default=None)
    args = parser.parse_args()

    with h5py.File(args.file_in, "r") as h5f:
        timeseries = np.array(h5f[args.timeseries_name])
        umean = np.array(h5f["u_mean"])

    if args.max is not None:
        timeseries = timeseries[:args.max, :]
        umean = umean[:args.max, :]

    file_out = (os.path.splitext(args.file_in)[0] + "_" +
                args.timeseries_name + ".png")

    plt.imsave(file_out,
               timeseries,
               origin='lower',
               cmap=plt.get_cmap('viridis'))

    uzmean = umean[:, 2]
    dzmean = uzmean * args.dt
    zmean = np.cumsum(dzmean)

    nt, nz = timeseries.shape
    dz = args.L/nz

    shifted_series = np.zeros_like(timeseries)
    for it in xrange(nt):
        iz_first = int((zmean[it] % args.L)/dz)
        iz_last = nz-iz_first
        shifted_series[it, :iz_last] = timeseries[it, iz_first:]
        shifted_series[it, iz_last:] = timeseries[it, :iz_first]

    file_shifted_out = (os.path.splitext(args.file_in)[0] + "_" +
                        args.timeseries_name + "_shifted.png")

    plt.imsave(file_shifted_out,
               shifted_series,
               origin='lower',
               cmap=plt.get_cmap('viridis'))

    # plt.figure()
    # plt.plot(args.dt * np.arange(len(zmean)), zmean)
    # plt.show()

if __name__ == "__main__":
    main()
