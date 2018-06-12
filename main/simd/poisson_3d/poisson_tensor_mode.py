import dolfin
import dolfin.cpp
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector, PETScKrylovSolver
from dolfin.jit.jit import ffc_jit

import numba as nb
import numpy as np

import loopy as lp
import islpy as isl
import pymbolic.primitives as pb

import cffi
import importlib
import itertools

import os

import simd.utils as utils
from simd.generate_ref_tensor import generate_ref_tensor


# C code for Laplace operator tensor tabulation
TABULATE_C = """
void tabulate_tensor_A(double* A_T, const double* const* w, double* coords, int cell_orientation)
{
    // Compute cell geometry tensor G_T
    alignas(32) double G_T[9];
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

TABULATE_C_FFC = """#include <stdalign.h>
void tabulate_tensor_A(double* restrict A, const double* const* w,
                       const double* restrict coordinate_dofs,
                       int cell_orientation)
{
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [entities][points][dofs]
    // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
    // PM* dimensions: [entities][dofs][dofs]
    alignas(32) static const double FE8_C0_D001_Q1[1][1][2] = { { { -1.0, 1.0 } } };
    // Unstructured piecewise computations
    const double J_c4 = coordinate_dofs[1] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[7] * FE8_C0_D001_Q1[0][0][1];
    const double J_c8 = coordinate_dofs[2] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[11] * FE8_C0_D001_Q1[0][0][1];
    const double J_c5 = coordinate_dofs[1] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[10] * FE8_C0_D001_Q1[0][0][1];
    const double J_c7 = coordinate_dofs[2] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[8] * FE8_C0_D001_Q1[0][0][1];
    const double J_c0 = coordinate_dofs[0] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[3] * FE8_C0_D001_Q1[0][0][1];
    const double J_c1 = coordinate_dofs[0] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[6] * FE8_C0_D001_Q1[0][0][1];
    const double J_c6 = coordinate_dofs[2] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[5] * FE8_C0_D001_Q1[0][0][1];
    const double J_c3 = coordinate_dofs[1] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[4] * FE8_C0_D001_Q1[0][0][1];
    const double J_c2 = coordinate_dofs[0] * FE8_C0_D001_Q1[0][0][0] + coordinate_dofs[9] * FE8_C0_D001_Q1[0][0][1];
    alignas(32) double sp[80];
    sp[0] = J_c4 * J_c8;
    sp[1] = J_c5 * J_c7;
    sp[2] = sp[0] + -1 * sp[1];
    sp[3] = J_c0 * sp[2];
    sp[4] = J_c5 * J_c6;
    sp[5] = J_c3 * J_c8;
    sp[6] = sp[4] + -1 * sp[5];
    sp[7] = J_c1 * sp[6];
    sp[8] = sp[3] + sp[7];
    sp[9] = J_c3 * J_c7;
    sp[10] = J_c4 * J_c6;
    sp[11] = sp[9] + -1 * sp[10];
    sp[12] = J_c2 * sp[11];
    sp[13] = sp[8] + sp[12];
    sp[14] = sp[2] / sp[13];
    sp[15] = J_c3 * (-1 * J_c8);
    sp[16] = sp[4] + sp[15];
    sp[17] = sp[16] / sp[13];
    sp[18] = sp[11] / sp[13];
    sp[19] = sp[14] * sp[14];
    sp[20] = sp[14] * sp[17];
    sp[21] = sp[18] * sp[14];
    sp[22] = sp[17] * sp[17];
    sp[23] = sp[18] * sp[17];
    sp[24] = sp[18] * sp[18];
    sp[25] = J_c2 * J_c7;
    sp[26] = J_c8 * (-1 * J_c1);
    sp[27] = sp[25] + sp[26];
    sp[28] = sp[27] / sp[13];
    sp[29] = J_c0 * J_c8;
    sp[30] = J_c6 * (-1 * J_c2);
    sp[31] = sp[29] + sp[30];
    sp[32] = sp[31] / sp[13];
    sp[33] = J_c1 * J_c6;
    sp[34] = J_c0 * J_c7;
    sp[35] = sp[33] + -1 * sp[34];
    sp[36] = sp[35] / sp[13];
    sp[37] = sp[28] * sp[28];
    sp[38] = sp[28] * sp[32];
    sp[39] = sp[28] * sp[36];
    sp[40] = sp[32] * sp[32];
    sp[41] = sp[32] * sp[36];
    sp[42] = sp[36] * sp[36];
    sp[43] = sp[37] + sp[19];
    sp[44] = sp[38] + sp[20];
    sp[45] = sp[39] + sp[21];
    sp[46] = sp[40] + sp[22];
    sp[47] = sp[41] + sp[23];
    sp[48] = sp[24] + sp[42];
    sp[49] = J_c1 * J_c5;
    sp[50] = J_c2 * J_c4;
    sp[51] = sp[49] + -1 * sp[50];
    sp[52] = sp[51] / sp[13];
    sp[53] = J_c2 * J_c3;
    sp[54] = J_c0 * J_c5;
    sp[55] = sp[53] + -1 * sp[54];
    sp[56] = sp[55] / sp[13];
    sp[57] = J_c0 * J_c4;
    sp[58] = J_c1 * J_c3;
    sp[59] = sp[57] + -1 * sp[58];
    sp[60] = sp[59] / sp[13];
    sp[61] = sp[52] * sp[52];
    sp[62] = sp[52] * sp[56];
    sp[63] = sp[60] * sp[52];
    sp[64] = sp[56] * sp[56];
    sp[65] = sp[60] * sp[56];
    sp[66] = sp[60] * sp[60];
    sp[67] = sp[43] + sp[61];
    sp[68] = sp[44] + sp[62];
    sp[69] = sp[45] + sp[63];
    sp[70] = sp[46] + sp[64];
    sp[71] = sp[47] + sp[65];
    sp[72] = sp[48] + sp[66];
    sp[73] = fabs(sp[13]);
    sp[74] = sp[67] * sp[73];
    sp[75] = sp[68] * sp[73];
    sp[76] = sp[69] * sp[73];
    sp[77] = sp[70] * sp[73];
    sp[78] = sp[71] * sp[73];
    sp[79] = sp[72] * sp[73];
    A[0] = 0.1666666666666667 * sp[74] + 0.1666666666666667 * sp[75] + 0.1666666666666667 * sp[76] + 0.1666666666666667 * sp[75] + 0.1666666666666667 * sp[77] + 0.1666666666666667 * sp[78] + 0.1666666666666667 * sp[76] + 0.1666666666666667 * sp[78] + 0.1666666666666667 * sp[79];
    A[1] = -0.1666666666666667 * sp[74] + -0.1666666666666667 * sp[75] + -0.1666666666666667 * sp[76];
    A[2] = -0.1666666666666667 * sp[75] + -0.1666666666666667 * sp[77] + -0.1666666666666667 * sp[78];
    A[3] = -0.1666666666666667 * sp[76] + -0.1666666666666667 * sp[78] + -0.1666666666666667 * sp[79];
    A[4] = -0.1666666666666667 * sp[74] + -0.1666666666666667 * sp[75] + -0.1666666666666667 * sp[76];
    A[5] = 0.1666666666666667 * sp[74];
    A[6] = 0.1666666666666667 * sp[75];
    A[7] = 0.1666666666666667 * sp[76];
    A[8] = -0.1666666666666667 * sp[75] + -0.1666666666666667 * sp[77] + -0.1666666666666667 * sp[78];
    A[9] = 0.1666666666666667 * sp[75];
    A[10] = 0.1666666666666667 * sp[77];
    A[11] = 0.1666666666666667 * sp[78];
    A[12] = -0.1666666666666667 * sp[76] + -0.1666666666666667 * sp[78] + -0.1666666666666667 * sp[79];
    A[13] = 0.1666666666666667 * sp[76];
    A[14] = 0.1666666666666667 * sp[78];
    A[15] = 0.1666666666666667 * sp[79];
}
"""


# C header for Laplace operator tensor tabulation
TABULATE_H = """
void tabulate_tensor_A(double* A, const double* const* w, double* coords, int cell_orientation);
"""


def reference_tensor(element: FiniteElement):
    """Generates code for the Laplace P1(Tetrahedron) reference tensor"""

    # Generate the reference tensor
    A0 = generate_ref_tensor(element)
    # Eliminate negative zeros
    A0[A0 == 0] = 0

    n_dof = A0.shape[0]
    n_dim = A0.shape[2]

    assert (A0.shape[0] == A0.shape[1])
    assert (A0.shape[2] == A0.shape[3])

    A0_flat = A0.reshape((n_dof**2, n_dim**2))
    non_zeros = A0_flat != 0
    nnz = np.sum(non_zeros, axis=1)
    index_ptrs = np.tile(np.arange(n_dim**2), (n_dof**2,1))[non_zeros]
    vals = A0_flat[non_zeros]

    vals_string = f"alignas(32) static const double A0_entries[{vals.size}] = {{\n#\n}};".replace("#", ", ".join(str(x) for x in vals))
    nnz_string = f"static const int A0_nnz[{nnz.size}] = {{\n#\n}};".replace("#", ", ".join(str(x) for x in nnz))
    ptrs_string = f"static const int A0_idx[{index_ptrs.size}] = {{\n#\n}};".replace("#", ", ".join(str(x) for x in index_ptrs))

    A0_string = f"alignas(32) static const double A0[{A0.size}] = {{\n#\n}};"
    numbers = ",\n".join(", ".join(str(x) for x in A0[i,j,:,:].flatten()) for i,j in itertools.product(range(n_dof),range(n_dof)))
    A0_string = A0_string.replace("#", numbers)

    return "\n".join([f"#define A0_NNZ {vals.size}", nnz_string, vals_string, ptrs_string, A0_string]), n_dof, n_dim


class cffi_kernels:
    @staticmethod
    def kernel_manual():
        """Handwritten cell kernel for the Laplace operator"""

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

    @staticmethod
    def kernel_avx(knl_name: str):
        code = """
    #include <immintrin.h>
    void {knl_name}(double *restrict A_T, 
                         double const *restrict A0, 
                         double const *restrict G_T)
    {
        for (int i = 0; i <= 3; ++i) {
            for (int j = 0; j <= 3; ++j) {
                __m256d g1 = _mm256_load_pd(&G_T[0]);
                __m256d a1 = _mm256_load_pd(&A0[36 * i + 9 * j + 0]);
                __m256d res1 = _mm256_mul_pd(g1, a1);
    
                __m256d g2 = _mm256_load_pd(&G_T[4]);
                __m256d a2 = _mm256_load_pd(&A0[36 * i + 9 * j + 4]);
                __m256d res2 = _mm256_fmadd_pd(g2, a2, res1);
                __m256d res3 = _mm256_hadd_pd(res2, res2);
    
                A_T[4 * i + j] = ((double*)&res3)[0] + ((double*)&res3)[2] + A0[36 * i + 9 * j + 8]*G_T[8];
            }
        }    
    }""".replace("{knl_name}", knl_name)

        header = """void {knl_name}(double *restrict A_T, 
                                        double const *restrict A0, 
                                        double const *restrict G_T);
            """.replace("{knl_name}", knl_name)

        return code, header

    @staticmethod
    def kernel_broadcast(knl_name: str):
        code = """
    #include <immintrin.h>
    void {knl_name}(double *restrict A_T, 
                         double const *restrict A0_entries,
                         int const *restrict A0_idx,
                         int const *restrict A0_nnz, 
                         double const *restrict G_T)
    {      
        // Multiply
        alignas(32) double A_T_scattered[A0_NNZ];
        for (int i = 0; i < A0_NNZ; i+=4) {
            __m256d a0 = _mm256_load_pd(&A0_entries[i]);
            __m256d g = _mm256_set_pd(G_T[A0_idx[i+3]], 
                                      G_T[A0_idx[i+2]], 
                                      G_T[A0_idx[i+1]], 
                                      G_T[A0_idx[i]]);
            __m256d res = _mm256_mul_pd(a0, g);
            _mm256_store_pd(&A_T_scattered[i], res);
        }
        
        // Reduce
        double acc_A;
        double* curr_val = &A_T_scattered[0];
        for (int i = 0; i < 16; ++i) {
            acc_A = 0;
            
            const double* end = curr_val + A0_nnz[i];
            for (; curr_val != end; ++curr_val) {
                acc_A += *(curr_val);
            }
            
            A_T[i] = acc_A;
        }
    }""".replace("{knl_name}", knl_name)

        header = """void {knl_name}(double *restrict A_T, 
                                    double const *restrict A0_entries,
                                    int const *restrict A0_idx,
                                    int const *restrict A0_nnz, 
                                    double const *restrict G_T);
        """.replace("{knl_name}", knl_name)

        return code, header

    @staticmethod
    def kernel_loopy(knl_name: str, n_dof: int, n_dim: int, verbose: bool = False):
        """Generate cell kernel for the Laplace operator using Loopy"""

        # Inputs to the kernel
        arg_names = ["A_T", "A0", "G_T"]
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

        if verbose:
            print("ISL loop domains:")
            print(isl_domains)
            print("")

        # Generate pymbolic variables for all used symbols
        args = {arg: pb.Variable(arg) for arg in arg_names}
        params = {param: pb.Variable(param) for param in param_names}
        inames = {iname: pb.Variable(iname) for iname, extent in loops}

        # Input arguments for the loopy kernel
        n, m = params["n"], params["m"]
        lp_args = {"A_T": lp.GlobalArg("A_T", dtype=np.double, shape=(n, n)),
                   "A0" : lp.GlobalArg("A0" , dtype=np.double, shape=(n, n, m)),
                   "G_T": lp.GlobalArg("G_T", dtype=np.double, shape=(m))}

        # Generate the list of arguments & parameters that will be passed to loopy
        data = []
        data += [arg for arg in lp_args.values()]
        data += [lp.ValueArg(param) for param in param_names]

        # Build the kernel instruction: computation and assignment of the element matrix
        def build_ass():
            #A_T[i,j] = sum(k, A0[i,j,k] * G_T[k]);

            # Get variable symbols for all required variables
            i,j,k = inames["i"], inames["j"], inames["k"]
            A_T, A0, G_T = args["A_T"], args["A0"], args["G_T"]

            # The target of the assignment
            target = pb.Subscript(A_T, (i, j))

            # The rhs expression: Frobenius inner product <A0[i,j],G_T>
            reduce_op = lp.library.reduction.SumReductionOperation()
            reduce_expr = pb.Subscript(A0, (i, j, k)) * pb.Subscript(G_T, (k))
            expr = lp.Reduction(reduce_op, k, reduce_expr)

            return lp.Assignment(target, expr)

        ass = build_ass()

        if verbose:
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

        knl = lp.fix_parameters(knl, n=n_dof, m=n_dim**2)
        knl = lp.prioritize_loops(knl, "i,j")

        if verbose:
            print("")
            print(knl)
            print("")

        # Generate kernel code
        knl_c, knl_h = lp.generate_code_v2(knl).device_code(), str(lp.generate_header(knl)[0])

        if verbose:
            print(knl_c)
            print("")

        # Postprocess kernel code
        knl_c = knl_c.replace("__restrict__", "restrict")
        knl_h = knl_h.replace("__restrict__", "restrict")

        return knl_c, knl_h


def compile_poisson_kernel(module_name: str, element: FiniteElement, verbose: bool = False):
    useFFCCode = False

    knl_name = "kernel_tensor_A"
    A0_code, n_dof, n_dim = reference_tensor(element)
    print("Computed reference tensor.")

    def useLoopy():
        print("Using Loopy generated kernel")
        knl_c, knl_h = cffi_kernels.kernel_loopy(knl_name, n_dof, n_dim, verbose)

        knl_call = f"{knl_name}(A_T, &A0[0], &G_T[0]);"
        knl_impl = knl_c + "\n"
        knl_sig = knl_h + "\n"

        return knl_call, knl_impl, knl_sig

    def useManualAVX():
        print("Using manual naive AVX kernel")
        knl_c, knl_h = cffi_kernels.kernel_avx(knl_name)

        # Use hand written kernel
        knl_call = f"{knl_name}(A_T, &A0[0], &G_T[0]);"
        knl_impl = knl_c + "\n"
        knl_sig = knl_h + "\n"

        return knl_call, knl_impl, knl_sig

    def useManualBroadcasted():
        print("Using manual 'broadcasted' AVX kernel")
        knl_c, knl_h = cffi_kernels.kernel_broadcast(knl_name)

        # Use hand written kernel
        knl_call = f"{knl_name}(A_T, &A0_entries[0], &A0_idx[0], &A0_nnz[0], &G_T[0]);"
        knl_impl = knl_c + "\n"
        knl_sig = knl_h + "\n"

        return knl_call, knl_impl, knl_sig

    knl_call, knl_impl, knl_sig = useLoopy()

    if useFFCCode:
        code_c = TABULATE_C_FFC
        code_h = TABULATE_H
    else:
        # Concatenate code of kernel and tabulate_tensor functions
        code_c = "\n".join(["#include <stdalign.h>\n", A0_code, knl_impl, TABULATE_C])
        # Insert code to execute kernel
        code_c = code_c.replace("{kernel}", knl_call)

        code_h = "\n".join([knl_sig, TABULATE_H])

    # Build the kernel
    ffi = cffi.FFI()
    ffi.set_source(module_name, code_c, extra_compile_args=["-O2",
                                                            "-march=native",
                                                            "-mtune=native"])
    ffi.cdef(code_h)
    ffi.compile(verbose=verbose)


class numba_kernels:
    @staticmethod
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

    @staticmethod
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


def generate_mesh(n):
    return UnitCubeMesh(MPI.comm_world, n, n, n)

    filename = "mesh.xdmf"
    if os.path.isfile(filename):
        with XDMFFile(MPI.comm_world, filename) as f:
            mesh = f.read_mesh(MPI.comm_world, dolfin.cpp.mesh.GhostMode.none)

        return mesh
    else:
        mesh = UnitCubeMesh(MPI.comm_world, n, n, n)

        with XDMFFile(MPI.comm_world, filename) as f:
            f.write(mesh, XDMFFile.Encoding.HDF5)

        return mesh


def solve():
    # Whether to use custom kernels instead of FFC
    useCustomKernels = True

    # Generate a unit cube with (n+1)^3 vertices
    n = 22
    mesh = None
    def get_mesh():
        nonlocal mesh
        mesh = generate_mesh(n)

    times = utils.timing(1, get_mesh, warm_up=False)
    print(f"Time for mesh generation: {times[0]*1000}ms")

    element = FiniteElement("P", tetrahedron, 3)
    Q = FunctionSpace(mesh, element)

    u = TrialFunction(Q)
    v = TestFunction(Q)

    # Define the boundary: vertices where any component is in machine precision accuracy 0 or 1
    def boundary(x):
        return np.sum(np.logical_or(x < DOLFIN_EPS, x > 1.0 - DOLFIN_EPS), axis=1) > 0

    u0 = Constant(0.0)
    bc = DirichletBC(Q, u0, boundary)

    if useCustomKernels:
        # Initialize bilinear form and rhs
        a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
        L = dolfin.cpp.fem.Form([Q._cpp_object])

        # Signature of tabulate_tensor functions
        sig = nb.types.void(nb.types.CPointer(nb.types.double),
                            nb.types.CPointer(nb.types.CPointer(nb.types.double)),
                            nb.types.CPointer(nb.types.double), nb.types.intc)

        # Compile the python functions using Numba
        fnA = nb.cfunc(sig, cache=True, nopython=True)(numba_kernels.tabulate_tensor_A)
        fnL = nb.cfunc(sig, cache=True, nopython=True)(numba_kernels.tabulate_tensor_L)

        module_name = "_laplace_kernel"
        compile_poisson_kernel(module_name, element, verbose=False)
        print("Finished compiling kernel")

        # Import the compiled kernel
        kernel_mod = importlib.import_module(f"simd.{module_name}")
        ffi, lib = kernel_mod.ffi, kernel_mod.lib

        # Get pointer to the compiled function
        fnA_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "tabulate_tensor_A"))

        # Get pointers to Numba functions
        # fnA_ptr = fnA.address
        fnL_ptr = fnL.address

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

    # Callable that performs assembly of matrix
    assembly_callable = lambda : assembler.assemble(A, dolfin.cpp.fem.Assembler.BlockType.monolithic)

    # Get timings for assembly of matrix over several runs
    n_runs = 10
    time_avg, time_min, time_max = utils.timing(n_runs, assembly_callable)
    print(f"Timings for assembly (n={n_runs}) avg: {time_avg*1000}ms, min: {time_min*1000}ms, max: {time_max*1000}ms")

    # Assemble again to get correct results
    A = PETScMatrix()
    assembler.assemble(A, dolfin.cpp.fem.Assembler.BlockType.monolithic)
    assembler.assemble(b, dolfin.cpp.fem.Assembler.BlockType.monolithic)

    Anorm = A.norm(dolfin.cpp.la.Norm.frobenius)
    bnorm = b.norm(dolfin.cpp.la.Norm.l2)
    print(Anorm, bnorm)

    if useCustomKernels:
        # Norms obtained with FFC and n=22
        assert (np.isclose(Anorm, 182.0409307016099))
        #assert (np.isclose(bnorm, 0.005296019976723171))

    return

    comm = L.mesh().mpi_comm()
    solver = PETScKrylovSolver(comm)

    u = Function(Q)
    solver.set_operator(A)
    solver.solve(u.vector(), b)

    # Export result
    #file = XDMFFile(MPI.comm_world, "poisson_3d.xdmf")
    #file.write(u, XDMFFile.Encoding.HDF5)


def run_example():
    solve()