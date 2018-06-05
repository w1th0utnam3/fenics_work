import dolfin
import dolfin.cpp
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector, PETScKrylovSolver
from dolfin.jit.jit import ffc_jit

import numba as nb
import numpy as np

import cffi
import importlib
import itertools

import time


# C code for Poisson tensor tabulation
TABULATE_C = """
void tabulate_tensor_A(double* A_T, double** w, double* coords, int cell_orientation)
{
    // Reference tensor
    {reference_tensor}
    
    // Compute geometry tensor G_T
    double G_T[9];
    {
        typedef double CoordsMat[4][3];
        CoordsMat* coordinate_dofs = (CoordsMat*)coords;
    
        const double* x0 = &((*coordinate_dofs)[0][0]);
        const double* x1 = &((*coordinate_dofs)[1][0]);
        const double* x2 = &((*coordinate_dofs)[2][0]);
        const double* x3 = &((*coordinate_dofs)[3][0]);
    
        // Entries of transformation matrix B
        const double a = x1[0] - x0[0];
        const double b = x2[0] - x0[0];
        const double c = x3[0] - x0[0];
        const double d = x1[1] - x0[1];
        const double e = x2[1] - x0[1];
        const double f = x3[1] - x0[1];
        const double g = x1[2] - x0[2];
        const double h = x2[2] - x0[2];
        const double i = x3[2] - x0[2];
    
        // Entries of inverse, transposed transformation matrix
        const double inv[9] = {
              (e*i - f*h),
             -(d*i - f*g),
              (d*h - e*g),
             -(b*i - c*h),
              (a*i - c*g),
             -(a*h - b*g),
              (b*f - c*e),
             -(a*f - c*d),
              (a*e - b*d)
        };
    
        const double detB = fabs(a*inv[0] + b*inv[1] + c*inv[2]);
        const double detB_inv2 = 1.0/(detB*detB);
        
        // G_T = Binv*Binv^T
        {
            double acc_G;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    acc_G = 0;
                    for (int k = 0; k < 3; ++k) {
                        acc_G += inv[i + k*3]*inv[j + k*3];
                    }
                    G_T[i*3 + j] = detB*detB_inv2*acc_G;
                }
            }
        }
    }
    
    // Apply kernel
    {kernel}

    return;
}
"""


# C header for Poisson tensor tabulation
TABULATE_H = """
void tabulate_tensor_A(double* A, double** w, double* coords, int cell_orientation);
"""

def reference_tensor():
    gradPhi = np.zeros((4, 3), dtype=np.double)
    gradPhi[0, :] = [-1, -1, -1]
    gradPhi[1, :] = [1, 0, 0]
    gradPhi[2, :] = [0, 1, 0]
    gradPhi[3, :] = [0, 0, 1]

    A0 = np.zeros((4,4,3,3), dtype=np.double)
    for i in range(4):
        for j in range(4):
            A0[i,j,:,:] = (1.0/6.0)*np.outer(gradPhi[i,:], gradPhi[j,:])

    # Eliminate negative zeros
    A0[A0 == 0] = 0

    A0_string = f"static const double A0[{A0.size}] = {{\n#\n}};"
    numbers = ",\n".join([", ".join([str(x) for x in A0[i,j,:,:].flatten()]) for i,j in itertools.product(range(4),range(4))])
    A0_string = A0_string.replace("#", numbers)

    return A0_string


def kernel():
    code = """{
        double acc_knl;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                acc_knl = 0;
                for (int k = 0; k < 9; ++k) {
                    acc_knl += A0[i*4*9 + j*9 + k] * G_T[k];
                }
                A_T[i*4 + j] = acc_knl;
            }
        }
    }"""
    return code


def compile_poisson_kernel(module_name: str, verbose: bool = False):
    code_c = TABULATE_C
    code_c = code_c.replace("{reference_tensor}", reference_tensor())
    code_c = code_c.replace("{kernel}", kernel())

    code_h = TABULATE_H

    # Build the kernel
    ffi = cffi.FFI()
    ffi.set_source(module_name, code_c, extra_compile_args=["-O2",
                                                            "-funroll-loops",
                                                            "-march=native",
                                                            "-mtune=native"])
    ffi.cdef(code_h)
    ffi.compile(verbose=verbose)


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

    A0 = np.zeros((4, 4, 3, 3), dtype=np.double)
    for i in range(4):
        for j in range(4):
            A0[i, j, :, :] = (1.0 / 6.0) * np.outer(gradPhi[i, :], gradPhi[j, :])

    G = np.abs(detB) * (Binv @ Binv.transpose())
    for i in range(4):
        for j in range(4):
            A[i, j] = np.sum(np.multiply(A0[i, j, :, :], G))


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
    n = 22
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

    module_name = "_laplace_kernel"
    compile_poisson_kernel(module_name, verbose=True)

    # Import the compiled kernel
    kernel_mod = importlib.import_module(module_name)
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    # Get pointer to the compiled function
    fnA_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "tabulate_tensor_A"))

    # Get pointers to Numba functions
    #fnA_ptr = fnA.address
    fnL_ptr = fnL.address

    if useCustomKernels:
        # Configure Forms to use own tabulate functions
        a.set_cell_tabulate(0, fnA_ptr)
        L.set_cell_tabulate(0, fnL_ptr)
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
    end = time.time()

    # We don't care about the RHS
    assembler.assemble(b, dolfin.cpp.fem.Assembler.BlockType.monolithic)

    print(f"Time for assembly: {(end-start)*1000.0}ms")

    Anorm = A.norm(dolfin.cpp.la.Norm.frobenius)
    bnorm = b.norm(dolfin.cpp.la.Norm.l2)
    print(Anorm, bnorm)

    # Norms obtained with FFC and n=13
    assert (np.isclose(Anorm, 60.86192203436385))
    assert (np.isclose(bnorm, 0.018075523965828778))

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