from __future__ import print_function
import argparse
import h5py
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruple probes.")
    parser.add_argument("probes_file", type=str, help="Input probes file.")
    args = parser.parse_args()

    with h5py.File(args.probes_file, "r") as h5f:
        coords = np.array(h5f["x"])

    N = len(coords)
    Nx = len(np.unique(coords[:, 0]))
    Ny = len(np.unique(coords[:, 1]))
    Nz = N//(Nx*Ny)
    print(Nx, Ny, Nz)
    
    x = coords[:, 0].reshape((Nx, Ny, Nz))
    y = coords[:, 1].reshape((Nx, Ny, Nz))
    z = coords[:, 2].reshape((Nx, Ny, Nz))

    dx = np.unique(coords[:, 0])
    dx = dx[1]-dx[0]
    x2 = x + x.max()-x.min() + dx

    dy = np.unique(coords[:, 1])
    dy = dy[1]-dy[0]
    y2 = y + y.max()-x.min() + dy

    X = np.zeros((2*Nx, 2*Ny, Nz))
    Y = np.zeros((2*Nx, 2*Ny, Nz))
    Z = np.zeros((2*Nx, 2*Ny, Nz))

    X[:Nx, :Ny, :] = x
    Y[:Nx, :Ny, :] = y
    Z[:Nx, :Ny, :] = z

    X[Nx:, :Ny, :] = x2
    Y[Nx:, :Ny, :] = y
    Z[Nx:, :Ny, :] = z

    X[:Nx, Ny:, :] = x
    Y[:Nx, Ny:, :] = y2
    Z[:Nx, Ny:, :] = z

    X[Nx:, Ny:, :] = x2
    Y[Nx:, Ny:, :] = y2
    Z[Nx:, Ny:, :] = z

    shape = (4*len(coords[:, 0]), 1)
    R = np.hstack((
        X.reshape(shape),
        Y.reshape(shape),
        Z.reshape(shape)))

    quad_probes_name = args.probes_file.split(".h")[0] + "_quad.h5"

    with h5py.File(quad_probes_name, "w") as h5f_out:
        h5f_out["x"] = R
