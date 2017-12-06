import h5py
from mpi4py import MPI
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri


def compute_volumes(node, elem, h5f):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    T = np.ones((4, 4))
    elem_vol = h5f.create_dataset("data/elem_vol", (len(elem),))
    node_ids = np.zeros(4, dtype=int)
    node_vol_proc = np.zeros(len(node))
    for i in xrange(rank, len(elem), size):
        node_ids[:] = elem[i, :]
        T[:, 1:] = node[node_ids]
        elem_vol[i] = np.abs(np.linalg.det(T)/6)
        node_vol_proc[node_ids] += elem_vol[i]/4
        if i % 10000 == 0:
            print i

    comm.Barrier()

    if rank == 0:
        node_vol = np.zeros_like(node_vol_proc)
    else:
        node_vol = None

    comm.Reduce(
        [node_vol_proc, MPI.DOUBLE],
        [node_vol, MPI.DOUBLE],
        op=MPI.SUM, root=0)

    dset_node_vol = h5f.create_dataset(
        "data/node_vol", (len(node),))
    if rank == 0:
        dset_node_vol[:] = node_vol


def compute_areas(node, face_wall, h5f):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    node_area_proc = np.zeros(len(node))
    f = np.zeros(3, dtype=int)
    v = np.zeros((3, 3))
    for i in xrange(rank, len(face_wall), size):
        f[:] = face_wall[i, :]
        v[:, :] = node[f]
        f_area = 0.5*np.linalg.norm(
            np.cross(v[0, :]-v[2, :], v[1, :]-v[2, :]))
        node_area_proc[f] += f_area/3.

    if rank == 0:
        node_area = np.zeros_like(node_area_proc)
    else:
        node_area = None

    comm.Reduce(
        [node_area_proc, MPI.DOUBLE],
        [node_area, MPI.DOUBLE],
        op=MPI.SUM, root=0)

    dset_node_area = h5f.create_dataset(
        "data/node_area", (len(node),))
    if rank == 0:
        dset_node_area[:] = node_area


def load_mesh(h5f):
    node = np.array(h5f["mesh/coordinates"])
    elem = np.array(h5f["mesh/topology"], dtype=int)
    return node, elem


def add_to_dict(face_dict, new_face):
    face_identity = tuple(sorted(new_face))
    if face_identity in face_dict:
        face_dict[face_identity] += 1
    else:
        face_dict[face_identity] = 1


def add_list_to_dict(face_dict, new_faces):
    for new_face in new_faces:
        add_to_dict(face_dict, new_face)


def on_edge(v, x_min, x_max, dims):
    for dim in dims:
        if np.all(v[:, dim]-x_min[dim] == 0.):
            return True
        elif np.all(v[:, dim]-x_max[dim] == 0.):
            return True
    return False


def find_wall_faces(node, elem, x_min, x_max, pbc_dims):
    face_dict = dict()
    for i, row in enumerate(elem):
        if i % 10000 == 0:
            print i
        add_list_to_dict(face_dict, [row[[0, 1, 2]], row[[0, 1, 3]],
                                     row[[0, 2, 3]], row[[1, 2, 3]]])
    print "Finding unique faces"
    face = np.array([key for key, val
                     in face_dict.iteritems() if val == 1])

    face_pbc_ids = []
    for i, f in enumerate(face):
        v = node[f]
        if on_edge(v, x_min, x_max, pbc_dims):
            face_pbc_ids.append(i)

    face_wall_ids = list(set(range(len(face)))-set(face_pbc_ids))
    face_wall = face[face_wall_ids, :]
    return face_wall


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="Compute mesh properties.")
    parser.add_argument("filename_in", type=str, help="Input filename (HDF5)")
    parser.add_argument("filename_out", type=str,
                        help="Output filename (HDF5)")
    args = parser.parse_args()
    pbc_dims = [0, 1]

    with h5py.File(args.filename_in, "r",
                   driver='mpio', comm=comm) as h5f_in:
        node, elem = load_mesh(h5f_in)

    x_min = np.min(node, 0)
    x_max = np.max(node, 0)

    if rank == 0:
        print "Min:", x_min[pbc_dims], "Max:", x_max[pbc_dims]

    h5f_out = h5py.File(args.filename_out, "w",
                        driver='mpio', comm=comm)

    if rank == 0:
        print "Computing volumes"

    compute_volumes(node, elem, h5f_out)

    if rank == 0:
        face_wall = find_wall_faces(node, elem, x_min, x_max, pbc_dims)
    else:
        face_wall = None

    face_wall = comm.bcast(face_wall, root=0)

    if rank == 0:
        print "Computing areas"

    compute_areas(node, face_wall, h5f_out)

    h5f_out.close()

    if rank == 0:
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_trisurf(node[:, 0], node[:, 1], node[:, 2],
                        triangles=face_wall,
                        cmap=plt.cm.Spectral)
        plt.show()


if __name__ == "__main__":
    main()
