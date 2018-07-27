import dolfin
import dolfin.cpp
import dolfin.fem
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


def tabulate_tensor_b(b_, w_, coords_, cell_orientation):
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

    b[:] = 0.0*vol / 4.0


def assembly():
    # Whether to use custom kernels instead of FFC
    useCustomKernels = True

    # Generate a unit cube with (n+1)^3 vertices
    n = 20
    mesh = UnitCubeMesh(MPI.comm_world, n, n, n)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    u = TrialFunction(Q)
    v = TestFunction(Q)

    def boundary0(x):
        wrong = np.logical_or(x[:, 1] < DOLFIN_EPS, x[:, 1] > 1.0 - DOLFIN_EPS)
        one = np.logical_or(x[:, 0] < DOLFIN_EPS, x[:, 0] > 1.0 - DOLFIN_EPS)
        two = np.logical_or(x[:, 2] < DOLFIN_EPS, x[:, 2] > 1.0 - DOLFIN_EPS)

        return np.logical_and(np.logical_or(one, two), np.logical_not(wrong))

    def boundary1(x):
        return np.logical_or(x[:,1] < DOLFIN_EPS, x[:,1] > 1.0 - DOLFIN_EPS)

    u0 = Constant(0.0)
    bc0 = DirichletBC(Q, u0, boundary0)

    u1 = Constant(1.0)
    bc1 = DirichletBC(Q, u1, boundary1)

    # Initialize bilinear form and rhs
    a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
    L = dolfin.cpp.fem.Form([Q._cpp_object])

    # Signature of tabulate_tensor functions
    sig = nb.types.void(nb.types.CPointer(nb.types.double),
                        nb.types.CPointer(nb.types.CPointer(nb.types.double)),
                        nb.types.CPointer(nb.types.double), nb.types.intc)

    # Compile the numba functions
    fnA = nb.cfunc(sig, cache=True, nopython=True)(tabulate_tensor_A)
    fnb = nb.cfunc(sig, cache=True, nopython=True)(tabulate_tensor_b)

    if useCustomKernels:
        # Configure Forms to use own tabulate functions
        a.set_cell_tabulate(0, fnA.address)
        L.set_cell_tabulate(0, fnb.address)
    else:
        # Use FFC
        jit_result = ffc_jit(dot(grad(u), grad(v)) * dx)
        ufc_form = dolfin.cpp.fem.make_ufc_form(jit_result[0])
        a = dolfin.cpp.fem.Form(ufc_form, [Q._cpp_object, Q._cpp_object])
        f = Expression("20.0", element=Q.ufl_element())
        jit_result = ffc_jit(f*v * dx)
        ufc_form = dolfin.cpp.fem.make_ufc_form(jit_result[0])
        L = dolfin.cpp.fem.Form(ufc_form, [Q._cpp_object])
        L.set_coefficient(0, f._cpp_object)

    start = time.time()
    assembler = dolfin.cpp.fem.Assembler([[a]], [L], [bc0, bc1])
    A = PETScMatrix()
    b = PETScVector()
    assembler.assemble(A, dolfin.cpp.fem.Assembler.BlockType.monolithic)
    assembler.assemble(b, dolfin.cpp.fem.Assembler.BlockType.monolithic)
    end = time.time()

    print(f"Time for assembly: {(end-start)*1000.0}ms")

    Anorm = A.norm(dolfin.cpp.la.Norm.frobenius)
    bnorm = b.norm(dolfin.cpp.la.Norm.l2)

    print(Anorm, bnorm)

    # Norms obtained with FFC and n=20
    #assert (np.isclose(Anorm, 55.82812911070811))
    #assert (np.isclose(bnorm, 29.73261456296761))

    comm = L.mesh().mpi_comm()
    solver = PETScKrylovSolver(comm)

    u = Function(Q)

    solver.set_operator(A)
    solver.solve(u.vector(), b)

    file = XDMFFile(MPI.comm_world, "poisson_3d.xdmf")
    file.write(u, XDMFFile.Encoding.HDF5)


def run_example():
    assembly()
