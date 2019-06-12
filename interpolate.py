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
    parser.add_argument('mesh_in', type=str,
                        help="mesh filename in (HDF5).")
    parser.add_argument('folder_in', type=str,
                        help="name of folder with velocity and"
                        " pressure (HDF5) file")
    parser.add_argument('ptsfilename', type=str,
                        help='name of points file (HDF5)')
    parser.add_argument('folder_out', type=str,
                        help='folder to save to (HDF5)')
    parser.add_argument('-id_first', type=int, help='id of first dataset',
                        default=0)
    parser.add_argument('-id_last', type=int, help='id of last dataset',
                        default=-1)
    parser.add_argument('-add', type=int, help='add number to output id',
                        default=0)
    parser.add_argument('-stress', action='store_true',
                        help='compute viscous stress')
    return parser.parse_args()


class Interpolation:
    def __init__(self, mesh_in, filename_in, folder_out=None,
                 p_filename_in=None, compute_stress=False):
        self.comm = mpi4py.MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()

        self.h5fmesh_str = mesh_in
        self.h5fu_str = filename_in
        self.h5fp_str = p_filename_in
        self.folder_out = folder_out

        if folder_out is not None and \
           not os.path.exists(folder_out) and self.mpi_rank == 0:
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

        self.compute_stress = compute_stress

        if self.compute_stress:
            df.info("Preparing to compute stresses")
            # self.Vv = df.VectorFunctionSpace(self.mesh, "DG", 0)
            self.Vv = df.VectorFunctionSpace(self.mesh, "CG", 1)
            # self.V2 = df.FunctionSpace(self.mesh, "CG", 2)
            self.grad_u = dict()
            for dim in range(3):
                self.grad_u[dim] = df.Function(self.Vv)

    def _make_dof_coords(self):
        dofmap = self.V.dofmap()
        my_first, my_last = dofmap.ownership_range()
        x = self.V.tabulate_dof_coordinates().reshape((-1, 3))
        unowned = dofmap.local_to_global_unowned()
        dofs = list(filter(lambda dof: dofmap.local_to_global_index(dof)
                           not in unowned,
                           range(my_last-my_first)))
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
        h5fmesh = df.HDF5File(self.mesh.mpi_comm(), self.h5fmesh_str, "r")
        h5fmesh.read(self.mesh, "/mesh", False)
        h5fmesh.close()

    def _initialize_geometry(self):
        """ Initialize from timeseries file """
        self._make_dof_coords()
        with h5py.File(self.h5fu_str, "r") as h5fu:
            self.x_data = np.array(h5fu.get("Mesh/0/mesh/geometry"))
        self._make_xdict(self.x_data)
        self.u = dict()
        for dim in range(3):
            self.u[dim] = df.Function(self.V)
        if self.h5fp_str is not None:
            self.p = df.Function(self.V)
        self.u_data = np.zeros_like(self.x_data)
        if self.h5fp_str is not None:
            self.p_data = np.zeros((len(self.x_data), 1))

    def set_probes(self, ptsfilename):
        df.info("Reading probe points")
        self.ptsfilename = ptsfilename
        with h5py.File(self.ptsfilename, "r") as h5fi:
            pts = np.array(h5fi["x"])
        self.set_probe_points(pts)

    def set_probe_points(self, pts):
        self.probes = StatisticsProbes(pts.flatten(), self.V, True)
        if self.h5fp_str is not None:
            self.probes_p = StatisticsProbes(pts.flatten(), self.V, False)
        if self.compute_stress:
            self.probes_grad_u = dict()
            for dim in range(3):
                self.probes_grad_u[dim] = StatisticsProbes(
                    pts.flatten(), self.Vv)

    def update(self, step):
        """ Update fields """
        with h5py.File(self.h5fu_str, "r") as h5fu:
            if step == "-1" or step == -1:
                steps = [int(key) for key in
                         h5fu["VisualisationVector"]]
                self.stepstr = str(np.max(steps))
            else:
                self.stepstr = step
            if "VisualisationVector/"+self.stepstr not in h5fu:
                return False
            self.u_data[:, :] = np.array(
                h5fu.get("VisualisationVector/"+self.stepstr))
        for dim in range(3):
            df.info("Setting u[" + str(dim) + "]")
            self._set_val(self.u[dim], self.u_data[:, dim])
        if self.h5fp_str is not None:
            with h5py.File(self.h5fp_str, "r") as h5fp:
                if "VisualisationVector/"+self.stepstr not in h5fp:
                    return False
                self.p_data[:] = np.array(
                    h5fp.get("VisualisationVector/"+self.stepstr))
                df.info("Setting p")
                self._set_val(self.p, self.p_data[:])
        if self.compute_stress:
            for dim in range(3):
                df.info("Computing grad(u[" + str(dim) + "])")
                # df.info(" --> step 1")
                # u2 = df.project(self.u[dim], self.V2,
                #                 solver_type="gmres",
                #                 preconditioner_type="amg",
                #                 form_compiler_parameters={"optimize": True})
                # df.info(" --> step 2")
                # self.grad_u[dim].assign(
                #    df.project(
                #         df.nabla_grad(u2), self.Vv,
                #         solver_type="gmres",
                #         preconditioner_type="amg",
                #         form_compiler_parameters={"optimize": True}))
                self.grad_u[dim].assign(
                    df.project(
                        df.nabla_grad(self.u[dim]), self.Vv,
                        solver_type="gmres",
                        preconditioner_type="amg",
                        form_compiler_parameters={"optimize": True}))
        return True

    def probe(self):
        self.probes.clear()
        df.info("Probing!")
        self.probes(self.u[0], self.u[1], self.u[2])
        if self.h5fp_str is not None:
            self.probes_p(self.p)
        if self.compute_stress:
            for dim in range(3):
                self.probes_grad_u[dim](self.grad_u[dim])

    def dump(self, add=0):
        data_mat = self.probes.array(1)
        if self.h5fp_str is not None:
            data_mat_p = self.probes_p.array(1)
        if self.compute_stress:
            data_mat_grad_u = dict()
            for dim in range(3):
                data_mat_grad_u[dim] = self.probes_grad_u[dim].array(1)
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
                if self.h5fp_str is not None:
                    h5f_out.create_dataset("/p/" + stepstrout,
                                           data=data_mat_p[:, 0])
                if self.compute_stress:
                    h5f_out.create_dataset(
                        "/grad_u/" + stepstrout,
                        data=np.hstack((data_mat_grad_u[0][:, :3],
                                        data_mat_grad_u[1][:, :3],
                                        data_mat_grad_u[2][:, :3])))

    def get_dsets(self, id_first, id_last):
        with h5py.File(self.h5fu_str, "r") as h5fu:
            dsets_u = set([int(key) for key in
                           h5fu["VisualisationVector/"].keys()])
        if self.h5fp_str is not None:
            with h5py.File(self.h5fp_str, "r") as h5fp:
                dsets_p = set([int(key) for key in
                               h5fp["VisualisationVector/"].keys()])
            dsets = [str(key) for key in sorted(list(dsets_u & dsets_p))]
        else:
            dsets = [str(key) for key in sorted(list(dsets_u))]
        if id_last >= len(dsets):
            id_last = len(dsets)-2
        if id_first > id_last:
            id_first = id_last
        dsets_loc = dsets[id_first:id_last]
        dsets_loc.append(dsets[id_last])
        return dsets_loc


def main():
    args = get_settings()
    u_filename = args.folder_in + "/u_from_tstep_0.h5"
    p_filename = args.folder_in + "/p_from_tstep_0.h5"
    interp = Interpolation(args.mesh_in, u_filename, args.folder_out,
                           p_filename_in=p_filename,
                           compute_stress=args.stress)
    dsets = interp.get_dsets(args.id_first, args.id_last)
    interp.set_probes(args.ptsfilename)
    for i_str in dsets:
        df.info("Step " + i_str)
        success = interp.update(i_str)
        if not success:
            break
        interp.probe()
        interp.dump(add=args.add)


if __name__ == "__main__":
    main()
