import dolfin as df
import fenicstools as ft
import argparse
import interpolate as itp
import mpi4py
from ufl.tensors import ListTensor


def import_mesh(filename):
    mesh_2 = df.Mesh()
    h5f_mesh_2 = df.HDF5File(mesh_2.mpi_comm(), filename, "r")
    h5f_mesh_2.read(mesh_2, "/mesh", False)
    h5f_mesh_2.close()
    return mesh_2


class AssignedVectorFunction(df.Function):
    """Vector function used for postprocessing.
    Assign data from ListTensor components using FunctionAssigner.
    """
    def __init__(self, u, name="Assigned Vector Function"):
        self.u = u
        assert isinstance(u, ListTensor)
        V = u[0].function_space()
        mesh = V.mesh()
        family = V.ufl_element().family()
        degree = V.ufl_element().degree()
        constrained_domain = V.dofmap().constrained_domain
        Vv = df.VectorFunctionSpace(mesh, family, degree,
                                    constrained_domain=constrained_domain)
        df.Function.__init__(self, Vv, name=name)
        self.fa = [df.FunctionAssigner(Vv.sub(i), V)
                   for i, _u in enumerate(u)]

    def __call__(self):
        for i, _u in enumerate(self.u):
            self.fa[i].assign(self.sub(i), _u)


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
    parser.add_argument("other_mesh_in", type=str, help="Other mesh file")
    parser.add_argument("initfile_out", type=str,
                        help="Output of initial file")
    args = parser.parse_args()
    return args


def main():
    args = get_settings()

    interp = itp.Interpolation(args.mesh_file_in,
                               args.u_file_in,
                               p_filename_in=args.p_file_in)
    interp.update(args.step)

    u_1 = interp.u
    p_1 = interp.p

    mesh_2 = import_mesh(args.other_mesh_in)
    S_2 = df.FunctionSpace(mesh_2, "CG", 1)

    u_ = dict()
    for key, val in u_1.iteritems():
        u_[key] = ft.interpolate_nonmatching_mesh(val, S_2)
    u__ = df.as_vector([u_[key] for key in u_.keys()])
    u = AssignedVectorFunction(u__)
    u()
    p = ft.interpolate_nonmatching_mesh(p_1, S_2)

    xdmff_u = df.XDMFFile(mesh_2.mpi_comm(), args.initfile_out + "_u.xdmf")
    xdmff_u.parameters["rewrite_function_mesh"] = False
    xdmff_u.parameters["flush_output"] = True
    xdmff_u.write(u, 0.)
    xdmff_p = df.XDMFFile(mesh_2.mpi_comm(), args.initfile_out + "_p.xdmf")
    xdmff_p.parameters["rewrite_function_mesh"] = False
    xdmff_p.parameters["flush_output"] = True
    xdmff_p.write(p, 0.)


if __name__ == "__main__":
    main()
