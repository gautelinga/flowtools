import dolfin as df


def main():
    Nx = 12
    Ny = Nx
    Nz = 40*Nx
    mesh = df.BoxMesh(df.Point(0., 0., 0.),
                      df.Point(1., 1., 40.),
                      Nx, Ny, Nz)
    h5f = df.HDF5File(mesh.mpi_comm(), "simple_duct.h5", "w")
    h5f.write(mesh, "mesh")
    h5f.close()
