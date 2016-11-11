import dolfin as df
import fenicstools as ft


def main():
    mesh = df.Mesh()
    h5f = df.HDF5File(mesh.mpi_comm(), "simple_duct.h5", "r")
    h5f.read(mesh, "mesh", False)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    P = df.FunctionSpace(mesh, "CG", 1)
    u_expr = df.Expression(("0.", "0.", expr_str(100)), degree=1)
    u = ft.interpolate_nonmatching_mesh(u_expr, V)
    u_avg = df.assemble(u.sub(2)*df.dx)/40.

    u.vector()[:] /= u_avg

    xdmff = df.XDMFFile(mesh.mpi_comm(), "simple_duct_u0.xdmf")
    xdmff.parameters["rewrite_function_mesh"] = False
    xdmff.parameters["flush_output"] = True
    xdmff.write(u, float(0.))

    p_expr = df.Expression("0.", degree=1)
    p = ft.interpolate_nonmatching_mesh(p_expr, P)

    xdmfp = df.XDMFFile(mesh.mpi_comm(), "simple_duct_p0.xdmf")
    xdmfp.parameters["rewrite_function_mesh"] = False
    xdmfp.parameters["flush_output"] = True
    xdmfp.write(p, float(0.))

    df.plot(u.sub(2), interactive=True)


def expr_str_n(n):
    nstr = str(n)
    return ("4./pow(" + nstr + "*pi,3)*(1.-cosh(" +
            nstr + "*pi*(x[0]-0.5))/cosh(" + nstr +
            "*pi*0.5))*sin(" + nstr + "*pi*x[1])")


def expr_str(n):
    string = expr_str_n(1)
    for i in xrange(3, n, 2):
        string += "+" + expr_str_n(i)
    return string


if __name__ == "__main__":
    main()
