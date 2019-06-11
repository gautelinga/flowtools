from __future__ import print_function
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


parser = argparse.ArgumentParser(description="Plot stuff seen from above.")
parser.add_argument("probes_file", type=str, help="Probes file")
parser.add_argument("folder", type=str, help="Data folder")
parser.add_argument("--noplot", action="store_true", help="Don't plot")
parser.add_argument("-i_start", type=int, default=None, help="First id")
parser.add_argument("-i_stop", type=int, default=None, help="Last id")
args = parser.parse_args()

with h5py.File(args.probes_file, "r") as h5f:
    coords = np.array(h5f["x"])

filenames = dict()
for filename in glob.glob(os.path.join(args.folder, "*.h5")):
    file_id = int(filename.split("u_")[-1].split(".h5")[0])
    if bool((args.i_start is None or file_id >= args.i_start) and
            (args.i_stop is None or file_id < args.i_stop)):
        filenames[file_id] = filename

data = []
for file_id in sorted(filenames.keys()):
    with h5py.File(filenames[file_id], "r") as h5f:
        print("File:", file_id, "of", len(filenames))
        u = np.array(h5f["u/"+str(file_id)])
        data.append(u)

do_plot = not args.noplot

N = len(coords)
Nx = len(np.unique(coords[:, 0]))
Ny = len(np.unique(coords[:, 1]))
Nz = N/(Nx*Ny)
print(Nx, Ny, Nz)

x = coords[:, 0].reshape((Nx, Ny, Nz))
y = coords[:, 1].reshape((Nx, Ny, Nz))
z = coords[:, 2].reshape((Nx, Ny, Nz))

u_mean = np.zeros_like(data[0])
du2 = np.zeros(len(data[0]))

u_avg_t = np.zeros((len(data), 4))
du2_t = np.zeros((len(data), 2))
for dset in data:
    u_mean[:, :] += dset[:, :]/len(data)
u_mean_avg = u_mean.mean(0)
for i, dset in enumerate(data):
    du2_i = np.sum((dset[:, :]-u_mean)**2, 1)
    du2[:] += du2_i/len(data)
    du2_t[i, 0] = i
    du2_t[i, 1] = du2_i.mean()
for i, dset in enumerate(data):
    u_avg_t[i, 0] = i
    u_avg_t[i, 1:] = dset[:, :].mean(0)

q = du2[:].reshape((Nx, Ny, Nz))

np.savetxt(os.path.join(args.folder,
                        "u_avg_t.dat"),
           u_avg_t)
np.savetxt(os.path.join(args.folder,
                        "du2_t.dat"),
           du2_t)

z_profile = np.mean(z, (0, 1))
du2_profile = np.mean(q, (0, 1))  # /w2.mean()
du2_plane = np.mean(q, 2)  # /w2.mean()

prefix = "du"

np.savetxt(os.path.join(args.folder,
                        prefix + "2_profile.dat"),
           np.vstack((z_profile,
                      du2_profile)).T)

np.savetxt(os.path.join(args.folder, prefix + "2_plane.dat"),
           du2_plane)

plt.figure()
plt.imshow(du2_plane, aspect=1.)
plt.colorbar()
plt.savefig(os.path.join(args.folder, prefix + "2.png"))

print("du2_mean =", du2.mean())

if do_plot:
    plt.show()
