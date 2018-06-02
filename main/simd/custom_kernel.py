import dolfin
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector
from dolfin.cpp.fem import SystemAssembler
from dolfin.jit.jit import ffc_jit

import numba as nb
import numpy as np

import cffi
import importlib


# C code for Poisson tensor tabulation
TABULATE_C = """
    void tabulate_tensor_A(double* A, double** w, double* coords, int cell_orientation)
    {
        typedef double CoordsMat[3][2];
        CoordsMat* coordinate_dofs = (CoordsMat*)coords;

        const double x0 = (*coordinate_dofs)[0][0];
        const double y0 = (*coordinate_dofs)[0][1];
        const double x1 = (*coordinate_dofs)[1][0];
        const double y1 = (*coordinate_dofs)[1][1];
        const double x2 = (*coordinate_dofs)[2][0];
        const double y2 = (*coordinate_dofs)[2][1];

        const double Ae = fabs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1));

        const double B[2][3] = {{y1-y2, y2-y0, y0-y1}, 
                                {x2-x1, x0-x2, x1-x0}};

        // A = (B^T B)/(2*Ae) 
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                A[i*3 + j] = 0.0;
                for (int k = 0; k < 2; k++) {
                    A[i*3 + j] += B[k][i] * B[k][j];
                } 
                A[i*3 + j] /= 2.0 * Ae;
            }
        }

        return;
    }

    void tabulate_tensor_b(double* b, double** w, double* coords, int cell_orientation)
    {
        typedef double CoordsMat[3][2];
        CoordsMat* coordinate_dofs = (CoordsMat*)coords;

        const double x0 = (*coordinate_dofs)[0][0];
        const double y0 = (*coordinate_dofs)[0][1];
        const double x1 = (*coordinate_dofs)[1][0];
        const double y1 = (*coordinate_dofs)[1][1];
        const double x2 = (*coordinate_dofs)[2][0];
        const double y2 = (*coordinate_dofs)[2][1];

        const double Ae = fabs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1));

        for (int i = 0; i < 3; i++) {
            b[i] = Ae / 6.0;
        }

        return;
    }
"""

# C header for Poisson tensor tabulation
TABULATE_H = """
    static void tabulate_tensor_A(double* A, double** w, double* coords, int cell_orientation);
    static void tabulate_tensor_b(double* b, double** w, double* coords, int cell_orientation);
"""


def compile_kernels(module_name: str, verbose: bool = False):
    ffi = cffi.FFI()

    ffi.set_source(module_name, TABULATE_C)
    ffi.cdef(TABULATE_H)
    lib = ffi.compile(verbose=verbose)

    return lib


def tabulate_tensor_A(A_, w_, coords_, cell_orientation):
    A = nb.carray(A_, (3, 3), dtype=np.float64)
    coordinate_dofs = nb.carray(coords_, (3, 2), dtype=np.float64)

    # Ke=∫Ωe BTe Be dΩ
    x0, y0 = coordinate_dofs[0, :]
    x1, y1 = coordinate_dofs[1, :]
    x2, y2 = coordinate_dofs[2, :]

    # 2x Element area Ae
    Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))

    B = np.array(
        [y1 - y2, y2 - y0, y0 - y1, x2 - x1, x0 - x2, x1 - x0],
        dtype=np.float64).reshape(2, 3)

    A[:, :] = (B.T @ B) / (2 * Ae)


def tabulate_tensor_b(b_, w_, coords_, cell_orientation):
    b = nb.carray(b_, (3), dtype=np.float64)
    coordinate_dofs = nb.carray(coords_, (6), dtype=np.float64)
    x0, y0 = coordinate_dofs[0*2 + 0], coordinate_dofs[0*2 + 1]
    x1, y1 = coordinate_dofs[1*2 + 0], coordinate_dofs[1*2 + 1]
    x2, y2 = coordinate_dofs[2*2 + 0], coordinate_dofs[2*2 + 1]

    # 2x Element area Ae
    Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    b[:] = Ae / 6.0


def assembly():
    # Whether to use custom kernels instead of FFC
    useCustomKernel = True
    # Whether to use CFFI kernels instead of Numba kernels
    useCffiKernel = False

    mesh = UnitSquareMesh(MPI.comm_world, 13, 13)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    u = TrialFunction(Q)
    v = TestFunction(Q)

    a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
    L = dolfin.cpp.fem.Form([Q._cpp_object])

    # Variant 1: Compile the Poisson kernel using CFFI
    kernel_name = "_poisson_kernel"
    compile_kernels(kernel_name)
    # Import the compiled kernel
    kernel_mod = importlib.import_module(kernel_name)
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    # Variant 2: Get pointers to the Numba kernels
    sig = nb.types.void(nb.types.CPointer(nb.types.double),
                        nb.types.CPointer(nb.types.CPointer(nb.types.double)),
                        nb.types.CPointer(nb.types.double), nb.types.intc)

    fnA = nb.cfunc(sig, cache=True, nopython=True)(tabulate_tensor_A)
    fnb = nb.cfunc(sig, cache=True, nopython=True)(tabulate_tensor_b)

    if useCustomKernel:

        if useCffiKernel:
            # Use the cffi kernel, compiled from raw C
            fnA_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "tabulate_tensor_A"))
            fnB_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "tabulate_tensor_b"))
        else:
            # Use the numba generated kernels
            fnA_ptr = fnA.address
            fnB_ptr = fnb.address

        a.set_cell_tabulate(0, fnA_ptr)
        L.set_cell_tabulate(0, fnB_ptr)

    else:
        # Use FFC
        ufc_form = ffc_jit(dot(grad(u), grad(v)) * dx)
        ufc_form = cpp.fem.make_ufc_form(ufc_form[0])
        a = cpp.fem.Form(ufc_form, [Q._cpp_object, Q._cpp_object])
        ufc_form = ffc_jit(v * dx)
        ufc_form = cpp.fem.make_ufc_form(ufc_form[0])
        L = cpp.fem.Form(ufc_form, [Q._cpp_object])

    assembler = cpp.fem.Assembler([[a]], [L], [])
    A = PETScMatrix()
    b = PETScVector()
    assembler.assemble(A, cpp.fem.Assembler.BlockType.monolithic)
    assembler.assemble(b, cpp.fem.Assembler.BlockType.monolithic)

    Anorm = A.norm(cpp.la.Norm.frobenius)
    bnorm = b.norm(cpp.la.Norm.l2)

    print(Anorm, bnorm)

    assert (np.isclose(Anorm, 56.124860801609124))
    assert (np.isclose(bnorm, 0.0739710713711999))

    #list_timings([TimingType.wall])

def run_example():
    assembly()
