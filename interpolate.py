from fenicstools import StatisticsProbes
import mpi4py
import h5py
import numpy as np
import dolfin as df
import argparse
import os


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
    parser = argparse.ArgumentParser(description="Interpolate to points.")
    parser.add_argument('filename_in', type=str,
                        help='name of velocity (HDF5) file')
    parser.add_argument('ptsfilename', type=str,
                        help='name of points (txt) file')
    parser.add_argument('folder_out', type=str,
                        help='folder to save to (HDF5)')
    parser.add_argument('id_first', type=int, help='id of first dataset')
    parser.add_argument('id_last', type=int, help='id of last dataset')
    parser.add_argument('-add', type=int, help='add number to output id',
                        default=0)
    return parser.parse_args()


class Interpolation:
    def __init__(self, filename_in, folder_out):
        self.comm = mpi4py.MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()

        self.h5fu_str = filename_in
        self.folder_out = folder_out

        if not os.path.exists(folder_out) and self.mpi_rank == 0:
            os.makedirs(folder_out)

        # Form compiler options
        df.parameters["form_compiler"]["optimize"] = True
        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters["form_compiler"]["representation"] = "quadrature"
        df.parameters["std_out_all_processes"] = False

        df.info("Loading mesh")
        self._load_mesh()

        df.info("Initializing elements and function spaces")
        E = df.FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        self.V = df.FunctionSpace(self.mesh, E)

        df.info("Initializing coordinates-to-node map")
        self._initialize_geometry()

    def _make_dof_coords(self):
        dofmap = self.V.dofmap()
        my_first, my_last = dofmap.ownership_range()
        x = self.V.tabulate_dof_coordinates().reshape((-1, 3))
        unowned = dofmap.local_to_global_unowned()
        dofs = filter(lambda dof: dofmap.local_to_global_index(dof)
                      not in unowned,
                      xrange(my_last-my_first))
        self.x = x[dofs]

    def _make_xdict(self, x_data):
        if self.mpi_rank == 0:
            self.xdict = dict([(tuple(x_list), i) for i, x_list in
                               enumerate(x_data.tolist())])
        else:
            self.xdict = None
        self.xdict = self.comm.bcast(self.xdict, root=0)

    def _set_val(self, f, f_data):
        vec = f.vector()
        values = vec.get_local()
        values[:] = [f_data[self.xdict[tuple(x_val)]]
                     for x_val in self.x.tolist()]
        vec.set_local(values)
        vec.apply('insert')

    def _load_mesh(self):
        """ Reads the mesh from the timeseries file """
        self.mesh = df.Mesh()
        h5fmesh = df.HDF5File(self.mesh.mpi_comm(), self.h5fu_str, "r")
        h5fmesh.read(self.mesh, "/Mesh/0", False)
        h5fmesh.close()

    def _initialize_geometry(self):
        """ Initialize from timeseries file """
        self._make_dof_coords()
        with h5py.File(self.h5fu_str, "r") as h5fu:
            self.x_data = np.array(h5fu.get("Mesh/0/coordinates"))
        self._make_xdict(self.x_data)
        self.u = dict()
        for dim in xrange(3):
            self.u[dim] = df.Function(self.V)

    def set_probes(self, ptsfilename):
        df.info("Reading probe points")
        self.ptsfilename = ptsfilename
        with h5py.File(self.ptsfilename, "r") as h5fi:
            pts = np.array(h5fi["x"])
        self.probes = StatisticsProbes(pts.flatten(), self.V, True)
        self.u_data = np.zeros_like(self.x_data)

    def update(self, step):
        """ Update fields """
        self.stepstr = step
        with h5py.File(self.h5fu_str, "r") as h5fu:
            if "VisualisationVector/"+self.stepstr not in h5fu:
                return False
            self.u_data[:, :] = np.array(
                h5fu.get("VisualisationVector/"+self.stepstr))
        for dim in xrange(3):
            df.info("Setting u[" + str(dim) + "]")
            self._set_val(self.u[dim], self.u_data[:, dim])
        return True

    def probe(self):
        self.probes.clear()
        df.info("Probing!")
        self.probes(self.u[0], self.u[1], self.u[2])

    def dump(self, add=0):
        data_mat = self.probes.array(1)
        if add == 0:
            stepstrout = self.stepstr
        else:
            stepstrout = str(int(self.stepstr) + add)
        if self.mpi_rank == 0:
            df.info("Saving!")
            with h5py.File(
                    os.path.join(self.folder_out, "u_" + stepstrout + ".h5"),
                    "w") as h5f_out:
                h5f_out.create_dataset("/u/" + stepstrout,
                                       data=data_mat[:, :3])


def main():
    args = get_settings()
    interp = Interpolation(args.filename_in, args.folder_out)
    interp.set_probes(args.ptsfilename)

    for i in xrange(args.id_first, args.id_last+1):
        i_str = str(i)

        df.info("Step " + i_str)
        success = interp.update(i_str)
        if not success:
            break
        interp.probe()
        interp.dump(add=args.add)


if __name__ == "__main__":
    main()
