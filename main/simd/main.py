import check_install

import numpy as np
import dolfin
from dolfin import *


def loopy_example():
    import numpy as np
    import loopy as lp
    import pyopencl as cl
    import pyopencl.array
    from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

    # setup
    # -----
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    n = 15 * 10 ** 6
    a = cl.array.arange(queue, n, dtype=np.float32)

    # create
    # ------
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]")

    # transform
    # ---------
    knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

    # execute
    # -------
    evt, (out,) = knl(queue, a=a)
    # ENDEXAMPLE

    knl = lp.add_and_infer_dtypes(knl, {"a": np.dtype(np.float32)})
    print(lp.generate_code_v2(knl).device_code())


def poisson():
    # Create mesh and define function space
    mesh = RectangleMesh.create(MPI.comm_world,
                                [Point(0, 0), Point(1, 1)], [32, 32],
                                CellType.Type.triangle, dolfin.cpp.mesh.GhostMode.none)
    V = FunctionSpace(mesh, "Lagrange", 1)

    cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return np.logical_or(x[:, 0] < DOLFIN_EPS, x[:, 0] > 1.0 - DOLFIN_EPS)

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("sin(5*x[0])", degree=2)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    #Assembler.assemble(a)

    # Compute solution
    u = Function(V)
    form = a == L
    solve(form, u, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Save solution in XDMF format
    #file = XDMFFile(MPI.comm_world, "poisson.xdmf")
    #file.write(u, XDMFFile.Encoding.HDF5)

    return


def main():
    if check_install.check("/local/fenics") != 0:
        print("Warning: Missing package was installed. Please rerun script.")
        return

    loopy_example()

    #poisson()
    return

if __name__ == '__main__':
    main()