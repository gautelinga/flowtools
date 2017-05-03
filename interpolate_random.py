import dolfin as df
import fenicstools as ft
import argparse
import interpolate as itp
import mpi4py
import numpy as np

__UINT32_MAX__ = np.iinfo('uint32').max


def import_mesh(filename):
    mesh_2 = df.Mesh()
    h5f_mesh_2 = df.HDF5File(mesh_2.mpi_comm(), filename, "r")
    h5f_mesh_2.read(mesh_2, "/mesh", False)
    h5f_mesh_2.close()
    return mesh_2


def get_settings():
    comm = mpi4py.MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    if mpi_rank == 0:
        args = get_settings_serial()
    else:
        args = None
    args = comm.bcast(args, root=0)
    return args

            
def get_settings_serial():
    parser = argparse.ArgumentParser(
        description="Interpolate velocity and pressure to another mesh.")
    parser.add_argument("mesh_file_in", type=str,
                        help="Mesh file in")
    parser.add_argument("u_file_in", type=str,
                        help="Velocity file in")
    parser.add_argument("p_file_in", type=str,
                        help="Pressure file in")
    parser.add_argument("--step", type=int, help="Timestep",
                        default=-1)
    parser.add_argument("-n", type=int,
                        help="number of randomly placed probes",
                        default=1000)
    parser.add_argument('-stress', action='store_true',
                        help='compute viscous stress')
    parser.add_argument("folder_out", type=str,
                        help="Filename out")
    args = parser.parse_args()
    return args


def main():
    args = get_settings()
    n_probes = args.n

    interp = itp.Interpolation(args.mesh_file_in,
                               args.u_file_in,
                               folder_out=args.folder_out,
                               p_filename_in=args.p_file_in,
                               compute_stress=args.stress)

    comm = mpi4py.MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    mesh = interp.mesh
    tree = mesh.bounding_box_tree()

    # element = interp.V.dolfin_element()
    # num_tensor_entries = element.value_dimension(0)
    # basis_matrix = np.zeros((element.space_dimension(),
    #                          num_tensor_entries))
    # coefficients = np.zeros(element.space_dimension())

    coords = mesh.coordinates()[:]
    coords_max_loc = np.max(coords, 0)
    coords_min_loc = np.min(coords, 0)
    coords_max = np.zeros_like(coords_max_loc)
    coords_min = np.zeros_like(coords_max_loc)
    comm.Allreduce(coords_max_loc, coords_max, op=mpi4py.MPI.MAX)
    comm.Allreduce(coords_min_loc, coords_min, op=mpi4py.MPI.MIN)

    if rank == 0:
        print "Min:", coords_min
        print "Max:", coords_max

    n_probes_outside = 0
    n_probes_inside = 0
    if rank == 0:
        x_probes = np.zeros((n_probes, 3))
        n_probe_stats = np.zeros((n_probes, 2), 'I')
    else:
        x_probes = None

    while n_probes_inside < n_probes:
        if rank == 0:
            x_probe = np.zeros(3)
            for dim in range(3):
                x_probe[dim] = np.random.uniform(coords_min[dim],
                                                 coords_max[dim])
        else:
            x_probe = None
        x_probe = comm.bcast(x_probe, root=0)

        point = df.Point(x_probe)
        coll = tree.compute_first_entity_collision(point)
        my_found = np.zeros(1, 'I')
        all_found = np.zeros(1, 'I')
        my_found[0] = not (coll == -1 or coll == __UINT32_MAX__)

        # Found on some process?
        comm.Allreduce(my_found, all_found)
        found = all_found[0]

        if found:
            if rank == 0:
                x_probes[n_probes_inside, :] = x_probe[:]
                n_probe_stats[n_probes_inside, :] = [n_probes_inside,
                                                     n_probes_outside]
                if n_probes_inside % 1000 == 0:
                    print n_probes_inside
            n_probes_inside += 1
        else:
            n_probes_outside += 1

    if rank == 0:
        print x_probes
        print n_probe_stats

        np.savetxt(args.folder_out + "/x_probes.dat", x_probes)
        np.savetxt(args.folder_out + "/n_probes.dat", n_probe_stats)
    x_probes = comm.bcast(x_probes, root=0)

    interp.set_probe_points(x_probes)

    interp.update(args.step)
    interp.probe()
    interp.dump()


if __name__ == "__main__":
    main()
