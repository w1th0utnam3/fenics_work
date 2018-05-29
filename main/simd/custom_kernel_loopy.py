import dolfin
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector
from dolfin.cpp.fem import SystemAssembler
from dolfin.jit.jit import ffc_jit

import numpy as np

import cffi
import importlib
import loopy as lp


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
                            
    kernel_tensor_A(A, &B[0][0], 1.0/(2.0*Ae));

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

    kernel_tensor_b(b, Ae / 6.0);

    return;
}
"""

# C header for Poisson tensor tabulation
TABULATE_H = """
void tabulate_tensor_A(double* A, double** w, double* coords, int cell_orientation);
void tabulate_tensor_b(double* b, double** w, double* coords, int cell_orientation);
"""


def replace_strings(text: str, replacement_pairs):
    new_text = text
    for pair in replacement_pairs:
        old, new = pair
        new_text = new_text.replace(old, new)
    return new_text


def build_kernel_A():
    knl = lp.make_kernel(
        "{ [i,j,k]: 0<=i,j,k<n }",
        """
            A[n*i + j] = x*sum(k, B[n*k + i]*B[n*k + j])
        """,
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        target=lp.CTarget())

    knl = lp.add_and_infer_dtypes(knl, {"A": np.dtype(np.double), "B": np.dtype(np.double), "x": np.dtype(np.double)})
    knl = lp.fix_parameters(knl, n=3)
    knl = lp.prioritize_loops(knl, "i,j")
    #print(knl)

    knl_c, knl_h = lp.generate_code_v2(knl).device_code(), str(lp.generate_header(knl)[0])

    replacements = [("loopy_kernel", "kernel_tensor_A"), ("__restrict__", "restrict")]
    knl_c = replace_strings(knl_c, replacements)
    knl_h = replace_strings(knl_h, replacements)

    return knl_c, knl_h


def build_kernel_b():
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        """
            b[i] = x
        """,
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        target=lp.CTarget())

    knl = lp.add_and_infer_dtypes(knl, {"b": np.dtype(np.double), "x": np.dtype(np.double)})
    knl = lp.fix_parameters(knl, n=3)
    #print(knl)

    knl_c, knl_h = lp.generate_code_v2(knl).device_code(), str(lp.generate_header(knl)[0])

    replacements = [("loopy_kernel", "kernel_tensor_b"), ("__restrict__", "restrict")]
    knl_c = replace_strings(knl_c, replacements)
    knl_h = replace_strings(knl_h, replacements)

    return knl_c, knl_h


def compile_kernels(module_name: str, verbose: bool = False):
    # Build loopy kernels
    knl_A_c, knl_A_h = build_kernel_A()
    knl_b_c, knl_b_h = build_kernel_b()

    # Build the module
    ffi = cffi.FFI()

    ffi.set_source(module_name, knl_A_c + "\n" + knl_b_c + "\n" + TABULATE_C)
    ffi.cdef(knl_A_h + knl_b_h + TABULATE_H)
    lib = ffi.compile(verbose=verbose)

    return lib


def assembly():
    # Whether to use custom kernels instead of FFC
    useCustomKernels = True

    mesh = UnitSquareMesh(MPI.comm_world, 13, 13)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    u = TrialFunction(Q)
    v = TestFunction(Q)

    a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
    L = dolfin.cpp.fem.Form([Q._cpp_object])

    # Compile the Poisson kernel using CFFI
    kernel_name = "_poisson_kernel"
    compile_kernels(kernel_name)
    # Import the compiled kernel
    kernel_mod = importlib.import_module(kernel_name)
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    # Get pointers to the CFFI functions
    fnA_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "tabulate_tensor_A"))
    fnB_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "tabulate_tensor_b"))

    # Configure Forms to use own tabulate functions
    a.set_cell_tabulate(0, fnA_ptr)
    L.set_cell_tabulate(0, fnB_ptr)

    if not useCustomKernels:
        # Use FFC
        ufc_form = ffc_jit(dot(grad(u), grad(v)) * dx)
        ufc_form = cpp.fem.make_ufc_form(ufc_form[0])
        a = cpp.fem.Form(ufc_form, [Q._cpp_object, Q._cpp_object])
        ufc_form = ffc_jit(v * dx)
        ufc_form = cpp.fem.make_ufc_form(ufc_form[0])
        L = cpp.fem.Form(ufc_form, [Q._cpp_object])

    assembler = cpp.fem.Assembler([[a]], [L], [])
    A = PETScMatrix(MPI.comm_world)
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
