import dolfin
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector
from dolfin.cpp.fem import SystemAssembler
from dolfin.jit.jit import ffc_jit

import numba as nb
import numpy as np

import cffi
import importlib
import loopy as lp

import islpy as isl
import pymbolic.primitives as pb

import utils
import time

# C code for Poisson tensor tabulation
TABULATE_C = """
void tabulate_tensor_A(double* A_mat, double** w, double* coords, int cell_orientation)
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
    
    // Invert B
    // Computation according to https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
    const double A =  (e*i - f*h);
    const double B = -(d*i - f*g);
    const double C =  (d*h - e*g);
    const double D = -(b*i - c*h);
    const double E =  (a*i - c*g);
    const double F = -(a*h - b*g);
    const double G =  (b*f - c*e);
    const double H = -(a*f - c*d);
    const double I =  (a*e - b*d);

    const double detB = a*A + b*B + c*C;
    const double detB_inv = 1.0/detB;

    const double Binv[3][3] = {{detB_inv*A, detB_inv*D, detB_inv*G}, 
                               {detB_inv*B, detB_inv*E, detB_inv*H},
                               {detB_inv*C, detB_inv*F, detB_inv*I}};
    
    static const double gradPhi[4][3] = {{-1, -1, -1}, 
                                         { 1,  0,  0},
                                         { 0,  1,  0},
                                         { 0,  0,  1}};
    
    // Bphi = G*B^-1
    double Bphi[4][3];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            Bphi[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
                Bphi[i][j] += gradPhi[i][k]*Binv[k][j];
            }
        }
    }
    
    // A = vol*Bphi*Bphi^T
    const double vol = (fabs(detB)/6.0);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            A_mat[4*i + j] = 0;
            for (int k = 0; k < 3; ++k) {
                A_mat[4*i + j] += vol*Bphi[i][k]*Bphi[j][k];
            }
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

    //kernel_tensor_b

    return;
}
"""

# C header for Poisson tensor tabulation
TABULATE_H = """
void tabulate_tensor_A(double* A, double** w, double* coords, int cell_orientation);
void tabulate_tensor_b(double* b, double** w, double* coords, int cell_orientation);
"""


def build_loopy_kernel_A_text():
    knl_name = "kernel_tensor_A"

    knl = lp.make_kernel(
        "{ [i,j,k]: 0<=i,j<n and 0<=k<m }",
        """
            A[i,j] = c*sum(k, B[k,i]*B[k,j])
        """,
        name=knl_name,
        assumptions="n >= 1 and m >= 1",
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        target=lp.CTarget())

    knl = lp.add_and_infer_dtypes(knl, {"A": np.dtype(np.double), "B": np.dtype(np.double), "c": np.dtype(np.double)})
    knl = lp.fix_parameters(knl, n=3, m=2)
    knl = lp.prioritize_loops(knl, "i,j")
    #print(knl)

    knl_c, knl_h = lp.generate_code_v2(knl).device_code(), str(lp.generate_header(knl)[0])

    replacements = [("__restrict__", "restrict")]
    knl_c = utils.replace_strings(knl_c, replacements)
    knl_h = utils.replace_strings(knl_h, replacements)

    knl_call = "kernel_tensor_A(A, &B[0][0], 1.0/(2.0*Ae));"

    return knl_name, knl_call, knl_c, knl_h


def build_loopy_kernel_A_auto():
    knl_name = "kernel_tensor_A"

    # Inputs to the kernel
    arg_names = ["A", "B", "c"]
    # Kernel parameters that will be fixed later
    param_names = ["n", "m"]
    # Tuples of inames and extents of their loops
    loops = [("i", "n"), ("j", "n"), ("k", "m")]

    # Generate the domains for the loops
    isl_domains = []
    for idx, extent in loops:
        # Create dict of loop variables (inames) and parameters
        vs = isl.make_zero_and_vars([idx], [extent])
        # Create the loop domain using '<=' and '>' restrictions
        isl_domains.append(((vs[0].le_set(vs[idx])) & (vs[idx].lt_set(vs[0] + vs[extent]))))

    print("ISL loop domains:")
    print(isl_domains)
    print("")

    # Generate pymbolic variables for all used symbols
    args = {arg : pb.Variable(arg) for arg in arg_names}
    params = {param : pb.Variable(param) for param in param_names}
    inames = {iname : pb.Variable(iname) for iname, extent in loops}

    # Input arguments for the loopy kernel
    lp_args = {"A": lp.GlobalArg("A", dtype=np.double, shape=(params["n"], params["n"])),
               "B": lp.GlobalArg("B", dtype=np.double, shape=(params["m"], params["n"])),
               "c": lp.ValueArg("c", dtype=np.double)}

    # Generate the list of arguments & parameters that will be passed to loopy
    data = []
    data += [arg for arg in lp_args.values()]
    data += [lp.ValueArg(param) for param in ["n", "m"]]

    # Build the kernel instruction: computation and assignment of the element matrix
    def build_ass():
        """
        A[i,j] = c*sum(k, B[k,i]*B[k,j])
        """

        # The target of the assignment
        target = pb.Subscript(args["A"], (inames["i"], inames["j"]))

        # The rhs expression: A reduce operation of the matrix columns
        # Maybe replace with manual increment?
        reduce_op = lp.library.reduction.SumReductionOperation()
        reduce_expr = pb.Subscript(args["B"], (inames["k"], inames["i"])) * pb.Subscript(args["B"], (inames["k"], inames["j"]))
        expr = args["c"]*lp.Reduction(reduce_op, inames["k"], reduce_expr)

        return lp.Assignment(target, expr)

    ass = build_ass()
    print("Assignment expression:")
    print(ass)
    print("")

    instructions = [ass]

    # Construct the kernel
    knl = lp.make_kernel(
        isl_domains,
        instructions,
        data,
        name=knl_name,
        target=lp.CTarget(),
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION)

    knl = lp.fix_parameters(knl, n=3, m=2)
    knl = lp.prioritize_loops(knl, "i,j")
    print(knl)
    print("")

    # Generate kernel code
    knl_c, knl_h = lp.generate_code_v2(knl).device_code(), str(lp.generate_header(knl)[0])
    print(knl_c)
    print("")

    # Postprocess kernel code
    replacements = [("__restrict__", "restrict")]
    knl_c = utils.replace_strings(knl_c, replacements)
    knl_h = utils.replace_strings(knl_h, replacements)

    knl_call = "kernel_tensor_A(A, &B[0][0], 1.0/(2.0*Ae));"

    return knl_name, knl_call, knl_c, knl_h


def build_loopy_kernel_b_text():
    knl_name = "kernel_tensor_b"

    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        """
            b[i] = c
        """,
        name="kernel_tensor_b",
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        target=lp.CTarget())

    knl = lp.add_and_infer_dtypes(knl, {"b": np.dtype(np.double), "c": np.dtype(np.double)})
    knl = lp.fix_parameters(knl, n=3)
    #print(knl)

    knl_c, knl_h = lp.generate_code_v2(knl).device_code(), str(lp.generate_header(knl)[0])

    replacements = [("__restrict__", "restrict")]
    knl_c = utils.replace_strings(knl_c, replacements)
    knl_h = utils.replace_strings(knl_h, replacements)

    knl_call = "kernel_tensor_b(b, Ae / 6.0);"

    return knl_name, knl_call, knl_c, knl_h


def build_manual_kernel_A():
    knl_name = "kernel_tensor_A"

    knl_c = ""
    knl_h = ""

    knl_call = """
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
    """
    knl_call = ""

    return knl_name, knl_call, knl_c, knl_h


def build_manual_kernel_b():
    knl_name = "kernel_tensor_b"

    knl_c = ""
    knl_h = ""

    knl_call = """
    for (int i = 0; i < 3; i++) {
            b[i] = Ae / 6.0;
    }
    """

    return knl_name, knl_call, knl_c, knl_h


def compile_kernels(module_name: str, verbose: bool = False):
    # Build loopy kernels
    knl_A_name, knl_A_call, knl_A_c, knl_A_h = build_manual_kernel_A()
    knl_b_name, knl_b_call, knl_b_c, knl_b_h = build_manual_kernel_b()

    # Glue together all parts of the assembly code
    assembly_c = knl_A_c + "\n" + knl_b_c + "\n" + TABULATE_C
    assembly_c = utils.replace_strings(assembly_c, [(f"//{knl_A_name}", knl_A_call), (f"//{knl_b_name}", knl_b_call)])
    assembly_h = knl_A_h + knl_b_h + TABULATE_H

    # Build the module
    ffi = cffi.FFI()
    ffi.set_source(module_name, assembly_c)
    ffi.cdef(assembly_h)
    lib = ffi.compile(verbose=verbose)

    return lib


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
    useCustomKernels = True

    mesh = UnitCubeMesh(MPI.comm_world, 13, 13, 13)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    u = TrialFunction(Q)
    v = TestFunction(Q)

    a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
    L = dolfin.cpp.fem.Form([Q._cpp_object])

    # Compile the Poisson kernel using CFFI
    kernel_name = "_poisson_kernel"
    compile_kernels(kernel_name, verbose=True)
    # Import the compiled kernel
    kernel_mod = importlib.import_module(kernel_name)
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    # Get pointers to the CFFI functions
    fnA_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "tabulate_tensor_A"))
    fnB_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "tabulate_tensor_b"))

    sig = nb.types.void(nb.types.CPointer(nb.types.double),
                        nb.types.CPointer(nb.types.CPointer(nb.types.double)),
                        nb.types.CPointer(nb.types.double), nb.types.intc)

    fnA = nb.cfunc(sig, cache=True, nopython=True)(tabulate_tensor_A)
    fnb = nb.cfunc(sig, cache=True, nopython=True)(tabulate_tensor_b)

    #fnA_ptr = fnA.address
    fnB_ptr = fnb.address

    if useCustomKernels:
        # Configure Forms to use own tabulate functions
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

    start = time.time()
    assembler = cpp.fem.Assembler([[a]], [L], [])
    A = PETScMatrix()
    b = PETScVector()
    assembler.assemble(A, cpp.fem.Assembler.BlockType.monolithic)
    assembler.assemble(b, cpp.fem.Assembler.BlockType.monolithic)
    end = time.time()

    print(f"Time for assembly: {(end-start)*1000.0}ms")

    Anorm = A.norm(cpp.la.Norm.frobenius)
    bnorm = b.norm(cpp.la.Norm.l2)

    print(Anorm, bnorm)

    #A_np = scipy2numpy(A.mat())

    assert (np.isclose(Anorm, 22.09531977181205))
    assert (np.isclose(bnorm, 0.020121054786640644))

    #list_timings([TimingType.wall])

def run_example():
    assembly()
