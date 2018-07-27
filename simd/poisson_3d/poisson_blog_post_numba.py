import dolfin
import dolfin.cpp
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector, PETScKrylovSolver
from dolfin.jit.jit import ffc_jit

import numba as nb
import numpy as np

import time


def tabulate_tensor_A(A_, w_, coords_, cell_orientation):
    '''Computes the Laplace cell tensor for linear 3D Lagrange elements'''

    A = nb.carray(A_, (4, 4), dtype=np.double)
    coordinate_dofs = nb.carray(coords_, (4, 3), dtype=np.double)

    # Coordinates of tet vertices
    x0 = coordinate_dofs[0, :]
    x1 = coordinate_dofs[1, :]
    x2 = coordinate_dofs[2, :]
    x3 = coordinate_dofs[3, :]

    # Reference to global transformation matrix
    B = np.zeros((3,3), dtype=np.double)
    B[:, 0] = x1 - x0
    B[:, 1] = x2 - x0
    B[:, 2] = x3 - x0

    Binv = np.linalg.inv(B)
    detB = np.linalg.det(B)

    # Matrix of basis function gradients
    gradPhi = np.zeros((4,3), dtype=np.double)
    gradPhi[0, :] = [-1, -1, -1]
    gradPhi[1, :] = [1, 0, 0]
    gradPhi[2, :] = [0, 1, 0]
    gradPhi[3, :] = [0, 0, 1]

    # Compute cell tensor
    Bphi = gradPhi @ Binv
    A[:,:] = np.abs(detB)/6.0*(Bphi @ Bphi.transpose())


def tabulate_tensor_L(b_, w_, coords_, cell_orientation):
    '''Computes the rhs for the Poisson problem with f=1 for linear 3D Lagrange elements'''

    b = nb.carray(b_, (4), dtype=np.float64)
    coordinate_dofs = nb.carray(coords_, (4, 3), dtype=np.double)

    # Coordinates of tet vertices
    x0 = coordinate_dofs[0, :]
    x1 = coordinate_dofs[1, :]
    x2 = coordinate_dofs[2, :]
    x3 = coordinate_dofs[3, :]

    # Reference to global transformation matrix
    B = np.zeros((3, 3), dtype=np.double)
    B[:, 0] = x1 - x0
    B[:, 1] = x2 - x0
    B[:, 2] = x3 - x0

    detB = np.linalg.det(B)
    vol = np.abs(detB)/6.0

    f = 2.0
    b[:] = f * (vol / 4.0)


def solve():
    # Whether to use custom Numba kernels instead of FFC
    useCustomKernels = True

    # Generate a unit cube with (n+1)^3 vertices
    n = 13
    mesh = UnitCubeMesh(MPI.comm_world, n, n, n)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    u = TrialFunction(Q)
    v = TestFunction(Q)

    # Define the boundary: vertices where any component is in machine precision accuracy 0 or 1
    def boundary(x):
        return np.sum(np.logical_or(x < DOLFIN_EPS, x > 1.0 - DOLFIN_EPS), axis=1) > 0

    u0 = Constant(0.0)
    bc = DirichletBC(Q, u0, boundary)

    # Initialize bilinear form and rhs
    a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
    L = dolfin.cpp.fem.Form([Q._cpp_object])

    # Signature of tabulate_tensor functions
    sig = nb.types.void(nb.types.CPointer(nb.types.double),
                        nb.types.CPointer(nb.types.CPointer(nb.types.double)),
                        nb.types.CPointer(nb.types.double), nb.types.intc)

    # Compile the python functions using Numba
    fnA = nb.cfunc(sig, cache=True, nopython=True)(tabulate_tensor_A)
    fnL = nb.cfunc(sig, cache=True, nopython=True)(tabulate_tensor_L)

    if useCustomKernels:
        # Configure Forms to use own tabulate functions
        a.set_cell_tabulate(0, fnA.address)
        L.set_cell_tabulate(0, fnL.address)
    else:
        # Use FFC

        # Bilinear form
        jit_result = ffc_jit(dot(grad(u), grad(v)) * dx)
        ufc_form = dolfin.cpp.fem.make_ufc_form(jit_result[0])
        a = dolfin.cpp.fem.Form(ufc_form, [Q._cpp_object, Q._cpp_object])

        # Rhs
        f = Expression("2.0", element=Q.ufl_element())
        jit_result = ffc_jit(f*v * dx)
        ufc_form = dolfin.cpp.fem.make_ufc_form(jit_result[0])
        L = dolfin.cpp.fem.Form(ufc_form, [Q._cpp_object])
        # Attach rhs expression as coefficient
        L.set_coefficient(0, f._cpp_object)

    assembler = dolfin.cpp.fem.Assembler([[a]], [L], [bc])
    A = PETScMatrix()
    b = PETScVector()

    # Perform assembly
    start = time.time()
    assembler.assemble(A, dolfin.cpp.fem.Assembler.BlockType.monolithic)
    assembler.assemble(b, dolfin.cpp.fem.Assembler.BlockType.monolithic)
    end = time.time()

    print(f"Time for assembly: {(end-start)*1000.0}ms")

    Anorm = A.norm(dolfin.cpp.la.Norm.frobenius)
    bnorm = b.norm(dolfin.cpp.la.Norm.l2)
    print(Anorm, bnorm)

    # Norms obtained with FFC and n=13
    assert (np.isclose(Anorm, 37.951697734708624))
    assert (np.isclose(bnorm, 0.03784180189499595))

    comm = L.mesh().mpi_comm()
    solver = PETScKrylovSolver(comm)

    u = Function(Q)
    solver.set_operator(A)
    solver.solve(u.vector(), b)

    # Export result
    file = XDMFFile(MPI.comm_world, "poisson_3d.xdmf")
    file.write(u, XDMFFile.Encoding.HDF5)


def run_example():
    solve()