import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


parser = argparse.ArgumentParser(description="Plot stuff seen from above.")
parser.add_argument("probes_file", type=str, help="Probes file")
parser.add_argument("folder", type=str, help="Data folder")
parser.add_argument("-m", "--mode", type=str, default="vorticity", help="Mode")
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
    print "File:", file_id, "of", len(filenames)
    with h5py.File(filenames[file_id], "r") as h5f:
        if args.mode == "vorticity":
            grad_u = np.array(h5f["grad_u/"+str(file_id)])
            omega = np.zeros((len(grad_u), 4))
            omega[:, 0] = grad_u[:, 7]-grad_u[:, 5]
            omega[:, 1] = grad_u[:, 2]-grad_u[:, 6]
            omega[:, 2] = grad_u[:, 3]-grad_u[:, 1]
            omega[:, 3] = omega[:, 0]**2 + omega[:, 1]**2 + omega[:, 2]**2
            data.append(omega)
        elif args.mode == "fluctuations":
            u = np.array(h5f["u/"+str(file_id)])
            data.append(u)

do_plot = not args.noplot

N = len(coords)
Nx = len(np.unique(coords[:, 0]))
Ny = len(np.unique(coords[:, 1]))
Nz = N/(Nx*Ny)
print Nx, Ny, Nz

x = coords[:, 0].reshape((Nx, Ny, Nz))
y = coords[:, 1].reshape((Nx, Ny, Nz))
z = coords[:, 2].reshape((Nx, Ny, Nz))

if args.mode == "vorticity":
    u = np.zeros_like(data[0])
    for dset in data:
        u[:, :] += dset[:, :]/len(data)

elif args.mode == "fluctuations":
    u_mean = np.zeros_like(data[0])
    u = np.zeros(len(data[0]))
    for dset in data:
        u_mean[:, :] += dset[:, :]/len(data)
    for dset in data:
        u[:] += np.sum((dset[:, :]-u_mean)**2, 1)/len(data)

if args.mode == "vorticity":
    w_x = u[:, 0].reshape((Nx, Ny, Nz))
    w_y = u[:, 1].reshape((Nx, Ny, Nz))
    w_z = u[:, 2].reshape((Nx, Ny, Nz))
    w_abs = np.sqrt(w_x**2 + w_y**2 + w_z**2)
    w_abs_mean = w_abs.mean()
    w2 = u[:, 3].reshape((Nx, Ny, Nz))

    w_x_mean = np.mean(w_x)
    w_x_profile = np.mean(w_x, (0, 1))/w_abs_mean
    w_y_profile = np.mean(w_y, (0, 1))/w_abs_mean
    w_z_profile = np.mean(w_z, (0, 1))/w_abs_mean
if args.mode == "fluctuations":
    w2 = u[:].reshape((Nx, Ny, Nz))

z_profile = np.mean(z, (0, 1))
w2_profile = np.mean(w2, (0, 1))/w2.mean()

if args.mode == "velocity":
    z_profile = np.insert(z_profile, 0, 0.)
    z_profile = np.append(z_profile, 1.)
    w_x_profile = np.insert(w_x_profile, 0, 0.)
    w_x_profile = np.append(w_x_profile, 0.)
    w_y_profile = np.insert(w_y_profile, 0, 0.)
    w_y_profile = np.append(w_y_profile, 0.)
    w_z_profile = np.insert(w_z_profile, 0, 0.)
    w_z_profile = np.append(w_z_profile, 0.)
    w2_profile = np.insert(w2_profile, 0, 0.)
    w2_profile = np.append(w2_profile, 0.)

prefix = args.mode

if args.mode == "vorticity":
    np.savetxt(os.path.join(args.folder,
                            prefix + "_profile.dat"),
               np.vstack((z_profile,
                          w_x_profile,
                          w_y_profile,
                          w_z_profile,
                          w2_profile)).T)

    w_x_plane = np.mean(w_x, 2)/w_abs_mean
    w_y_plane = np.mean(w_y, 2)/w_abs_mean
    w_z_plane = np.mean(w_z, 2)/w_abs_mean
    w_abs_plane = np.mean(w_abs, 2)/w_abs_mean
elif args.mode == "fluctuations":
    np.savetxt(os.path.join(args.folder,
                            prefix + "2_profile.dat"),
               np.vstack((z_profile,
                          w2_profile)).T)

w2_plane = np.mean(w2, 2)/w2.mean()

if args.mode == "vorticity":
    np.savetxt(os.path.join(args.folder, prefix + "_x_plane.dat"),
               w_x_plane)
    np.savetxt(os.path.join(args.folder, prefix + "_y_plane.dat"),
               w_y_plane)
    np.savetxt(os.path.join(args.folder, prefix + "_z_plane.dat"),
               w_z_plane)

np.savetxt(os.path.join(args.folder, prefix + "2_plane.dat"),
           w2_plane)

if args.mode == "vorticity":
    plt.figure()
    plt.imshow(w_x_plane, aspect=1.)
    plt.colorbar()
    plt.savefig(os.path.join(args.folder, prefix + "_x.png"))

    plt.figure()
    plt.imshow(w_y_plane, aspect=1.)
    plt.colorbar()
    plt.savefig(os.path.join(args.folder, prefix + "_y.png"))

    plt.figure()
    plt.imshow(w_z_plane, aspect=1.)
    plt.colorbar()
    plt.savefig(os.path.join(args.folder, prefix + "_z.png"))

    plt.figure()
    plt.imshow(w_abs_plane, aspect=1.)
    plt.colorbar()
    plt.savefig(os.path.join(args.folder, prefix + "_abs.png"))

plt.figure()
plt.imshow(w2_plane, aspect=1.)
plt.colorbar()
plt.savefig(os.path.join(args.folder, prefix + "2.png"))

print "w2_mean =", w2.mean()

if do_plot:
    plt.show()
