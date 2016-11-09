from fenicstools import StatisticsProbes
import mpi4py
import h5py
import numpy as np
import dolfin as df
import pandas as pd
import argparse
import os


comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description="Interpolate to points.")
parser.add_argument('filename_in', type=str,
                    help='name of velocity (HDF5) file')
parser.add_argument('pointsfilename', type=str,
                    help='name of points (txt) file')
parser.add_argument('folder_out', type=str, help='folder to save to (HDF5)')
parser.add_argument('id_first', type=int, help='id of first dataset')
parser.add_argument('id_last', type=int, help='id of last dataset')
args = parser.parse_args()

h5fu_str = args.filename_in
ptsfilename = args.pointsfilename
folder_out = args.folder_out

# Form compiler options
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "quadrature"
df.parameters["std_out_all_processes"] = False


def make_dof_coords(S):
    dofmap = S.dofmap()
    my_first, my_last = dofmap.ownership_range()
    x = S.tabulate_dof_coordinates().reshape((-1, 3))
    unowned = dofmap.local_to_global_unowned()
    dofs = filter(lambda dof: dofmap.local_to_global_index(dof)
                  not in unowned,
                  xrange(my_last-my_first))
    x = x[dofs]
    return x


def make_xdict(x_data):
    if rank == 0:
        xdict = dict([(tuple(x_list), i) for i, x_list in
                      enumerate(x_data.tolist())])
    else:
        xdict = None
    xdict = comm.bcast(xdict, root=0)
    return xdict


def set_val(f, f_data, x, xdict):
    vec = f.vector()
    values = vec.get_local()
    values[:] = [f_data[xdict[tuple(x_val)]] for x_val in x.tolist()]
    vec.set_local(values)
    vec.apply('insert')


def mesh():
    # Reads the mesh from the timeseries file
    mesh = df.Mesh()
    h5fmesh = df.HDF5File(mesh.mpi_comm(), h5fu_str, "r")
    h5fmesh.read(mesh, "/Mesh/0", False)
    h5fmesh.close()
    return mesh


def initialize(V):
    """ Initialize from timeseries file """
    x = make_dof_coords(V)
    with h5py.File(h5fu_str, "r") as h5fu:
        x_data = np.array(h5fu.get("Mesh/0/coordinates"))
    xdict = make_xdict(x_data)
    u0 = dict()
    for dim in xrange(3):
        u0[dim] = df.Function(V)
    df.info("Reading probe points")
    pts = np.asarray(pd.read_csv(ptsfilename, sep="\t", header=None))
    probes = StatisticsProbes(pts.flatten(), V, True)
    u_data = np.zeros_like(x_data)
    return u0, x, xdict, probes, u_data


def update(V, u, step, x, xdict, u_data):
    """ Update fields """
    with h5py.File(h5fu_str, "r") as h5fu:
        u_data[:, :] = np.array(h5fu.get("VisualisationVector/"+step))
    for dim in xrange(3):
        df.info("Setting u[" + str(dim) + "]")
        set_val(u[dim], u_data[:, dim], x, xdict)

if not os.path.exists(folder_out) and rank == 0:
    os.makedirs(folder_out)

df.info("Loading mesh")
msh = mesh()

df.info("Initializing elements and function spaces")
E = df.FiniteElement("Lagrange", msh.ufl_cell(), 1)
V = df.FunctionSpace(msh, E)

df.info("Initializing coordinates-to-node map")
u, x, xdict, probes, u_data = initialize(V)

for i in xrange(args.id_first, args.id_last+1):
    i_str = str(i)
    df.info("Step " + i_str)
    update(V, u, i_str, x, xdict, u_data)
    probes.clear()
    df.info("Probing!")
    probes(u[0], u[1], u[2])
    data_mat = probes.array(1)
    if rank == 0:
        df.info("Saving")
        with h5py.File(os.path.join(folder_out, "u_" + i_str + ".h5"),
                       "w") as h5f_out:
            h5f_out.create_dataset("/u/" + i_str, data=data_mat[:, :3])
