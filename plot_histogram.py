import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot histogram")
    parser.add_argument("filename", type=str, help="Input filename")
    parser.add_argument("datasetname", type=str, help="Input dataset")
    parser.add_argument("-c", type=int, help="Component",
                        default=0)
    parser.add_argument("-log", type=str, help="Log plot",
                        default="")
    parser.add_argument("-plt", action="store_true", help="Do plot")
    parser.add_argument("-save", type=str, default="", help="Save to folder")
    parser.add_argument("-nbins", type=int, default="25", help="Number of bins")
    args = parser.parse_args()

    datasetname = args.datasetname.split("/")[0]
    with h5py.File(args.filename, "r") as h5f:
        if np.array(h5f[args.datasetname]).ndim == 1:
            data = np.array(h5f[args.datasetname])
            outfilename = args.save + "/" + datasetname + ".hist"
        elif args.c >= 0:
            data = np.array(h5f[args.datasetname][:, args.c])
            outfilename = args.save + "/" + datasetname + "_" + \
                str(args.c) + ".hist"
        else:
            data_in = np.array(h5f[args.datasetname][:, :])
            data = np.sqrt(data_in[:, 0]**2 +
                           data_in[:, 1]**2 +
                           data_in[:, 2]**2)
            outfilename = args.save + "/" + datasetname + "_abs.hist"

    data_width = data.max()-data.min()
    nbins = args.nbins
    hist, bin_edges = np.histogram(data, bins=nbins, density=True,
                                   range=[data.min()-data_width/(2*nbins),
                                          data.max()+data_width/(2*nbins)])
    bins = 0.5*(bin_edges[1:]+bin_edges[:-1])

    if args.plt:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(bins, hist)
        if args.log == "x" or args.log == "xy" or args.log == "yx":
            ax.set_xscale('log')
        if args.log == "y" or args.log == "xy" or args.log == "yx":
            ax.set_yscale('log')
        plt.show()

    if args.save != "":
        np.savetxt(outfilename, np.vstack((bins, hist)).T)


if __name__ == "__main__":
    main()
