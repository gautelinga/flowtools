import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def main():
    parser = argparse.ArgumentParser(
        description="Plot timeseries.")
    parser.add_argument("file_in", type=str,
                        help="Timeseries file.")
    parser.add_argument("-u", type=str,
                        help="Mean velocity file.",
                        default=None)
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
    args = parser.parse_args()

    timeseries = np.loadtxt(args.file_in)

    file_out = os.path.splitext(args.file_in)[0] + ".png"

    plt.imsave(file_out,
               timeseries,
               origin='lower',
               cmap=plt.get_cmap('viridis'))

    if args.u is not None:
        umean = np.loadtxt(args.u)
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

        file_shifted_out = os.path.splitext(args.file_in)[0] + "_shifted.png"

        plt.imsave(file_shifted_out,
                   shifted_series,
                   origin='lower',
                   cmap=plt.get_cmap('viridis'))

        # plt.figure()
        # plt.plot(args.dt * np.arange(len(zmean)), zmean)
        # plt.show()

if __name__ == "__main__":
    main()
