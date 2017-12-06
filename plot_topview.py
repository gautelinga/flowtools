import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Plot stuff seen from above.")
parser.add_argument("probes_file", type=str, help="Probes file")
parser.add_argument("data_file", type=str, help="Data file")
parser.add_argument("-noplot", action="store_true", help="Don't plot")
args = parser.parse_args()

foutpre = args.data_file.split(".h")[0]

do_plot = not args.noplot

with h5py.File(args.data_file, "r") as h5f:
    u = np.array(h5f["u"][h5f["u"].keys()[-1]])

with h5py.File(args.probes_file, "r") as h5f:
    coords = np.array(h5f["x"])

# print u
# print x

i, j = 0, 0

N = len(coords)

# print coords.shape
Nx = len(np.unique(coords[:, 0]))
Ny = len(np.unique(coords[:, 1]))
Nz = N/(Nx*Ny)
print Nx, Ny, Nz

x = coords[:, 0].reshape((Nx, Ny, Nz))
y = coords[:, 1].reshape((Nx, Ny, Nz))
z = coords[:, 2].reshape((Nx, Ny, Nz))

u_x = u[:, 0].reshape((Nx, Ny, Nz))
u_y = u[:, 1].reshape((Nx, Ny, Nz))
u_z = u[:, 2].reshape((Nx, Ny, Nz))
u_abs = np.sqrt(u_x**2 + u_y**2 + u_z**2)

u_x_mean = np.mean(u_x)

z_profile = np.mean(z, (0, 1))
u_x_profile = np.mean(u_x, (0, 1))/u_x_mean
u_y_profile = np.mean(u_y, (0, 1))/u_x_mean
u_z_profile = np.mean(u_z, (0, 1))/u_x_mean

z_profile = np.insert(z_profile, 0, 0.)
z_profile = np.append(z_profile, 1.)
u_x_profile = np.insert(u_x_profile, 0, 0.)
u_x_profile = np.append(u_x_profile, 0.)
u_y_profile = np.insert(u_y_profile, 0, 0.)
u_y_profile = np.append(u_y_profile, 0.)
u_z_profile = np.insert(u_z_profile, 0, 0.)
u_z_profile = np.append(u_z_profile, 0.)

np.savetxt(foutpre + "_profile.dat", np.vstack(
    (z_profile, u_x_profile, u_y_profile, u_z_profile)).T)


u_x_plane = np.mean(u_x, 2)/u_x_mean
u_y_plane = np.mean(u_y, 2)/u_x_mean  # u_y.std()
u_z_plane = np.mean(u_z, 2)/u_x_mean  # u_z.std()
u_abs_plane = np.mean(u_abs, 2)/u_x_mean

np.savetxt(foutpre + "_u_x_plane.dat", u_x_plane)
np.savetxt(foutpre + "_u_y_plane.dat", u_y_plane)
np.savetxt(foutpre + "_u_z_plane.dat", u_z_plane)

plt.figure()
plt.imshow(u_x_plane, aspect=1.)
plt.colorbar()
plt.savefig(foutpre + "_u_x.png")

plt.figure()
plt.imshow(u_y_plane, aspect=1.)
plt.colorbar()
plt.savefig(foutpre + "_u_y.png")

plt.figure()
plt.imshow(u_z_plane, aspect=1.)
plt.colorbar()
plt.savefig(foutpre + "_u_z.png")

plt.figure()
plt.imshow(u_abs_plane, aspect=1.)
plt.colorbar()
plt.savefig(foutpre + "_u_abs.png")

if do_plot:
    plt.show()
