import dolfin
import dolfin.cpp
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector, PETScKrylovSolver
from dolfin.jit.jit import ffc_jit

import numba as nb
import numpy as np
import scipy.sparse as sps

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

# P1(Tet)
TABULATE_C_FFC_P1 = """#include <stdalign.h>
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

# P2(Tet)
TABULATE_C_FFC_P2 = """#include <stdalign.h>
void tabulate_tensor_A(
    double* restrict A, 
    const double* const* w,
    const double* restrict coordinate_dofs,
    int cell_orientation)
{
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [entities][points][dofs]
    // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
    // PM* dimensions: [entities][dofs][dofs]
    alignas(32) static const double FE9_C0_D001_Q4[1][1][2] = { { { -1.0, 1.0 } } };
    // Unstructured piecewise computations
    const double J_c4 = coordinate_dofs[1] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[7] * FE9_C0_D001_Q4[0][0][1];
    const double J_c8 = coordinate_dofs[2] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[11] * FE9_C0_D001_Q4[0][0][1];
    const double J_c5 = coordinate_dofs[1] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[10] * FE9_C0_D001_Q4[0][0][1];
    const double J_c7 = coordinate_dofs[2] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[8] * FE9_C0_D001_Q4[0][0][1];
    const double J_c0 = coordinate_dofs[0] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[3] * FE9_C0_D001_Q4[0][0][1];
    const double J_c1 = coordinate_dofs[0] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[6] * FE9_C0_D001_Q4[0][0][1];
    const double J_c6 = coordinate_dofs[2] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[5] * FE9_C0_D001_Q4[0][0][1];
    const double J_c3 = coordinate_dofs[1] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[4] * FE9_C0_D001_Q4[0][0][1];
    const double J_c2 = coordinate_dofs[0] * FE9_C0_D001_Q4[0][0][0] + coordinate_dofs[9] * FE9_C0_D001_Q4[0][0][1];
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
    A[0] = 0.1 * sp[74] + 0.1000000000000001 * sp[75] + 0.1 * sp[76] + 0.1000000000000001 * sp[75] + 0.1000000000000001 * sp[77] + 0.1000000000000001 * sp[78] + 0.1 * sp[76] + 0.1000000000000001 * sp[78] + 0.1000000000000001 * sp[79];
    A[1] = 0.03333333333333325 * sp[74] + 0.03333333333333331 * sp[75] + 0.0333333333333333 * sp[76];
    A[2] = 0.03333333333333339 * sp[75] + 0.03333333333333336 * sp[77] + 0.03333333333333338 * sp[78];
    A[3] = 0.03333333333333338 * sp[76] + 0.03333333333333338 * sp[78] + 0.03333333333333335 * sp[79];
    A[4] = 0.03333333333333456 * sp[75] + 0.03333333333333452 * sp[76] + 0.03333333333333455 * sp[77] + 0.03333333333333449 * sp[78] + 0.03333333333333448 * sp[78] + 0.03333333333333443 * sp[79];
    A[5] = 0.03333333333333458 * sp[74] + 0.03333333333333457 * sp[76] + 0.03333333333333457 * sp[75] + 0.03333333333333462 * sp[78] + 0.03333333333333448 * sp[76] + 0.03333333333333455 * sp[79];
    A[6] = 0.03333333333333456 * sp[74] + 0.0333333333333345 * sp[75] + 0.03333333333333453 * sp[75] + 0.03333333333333455 * sp[77] + 0.03333333333333448 * sp[76] + 0.03333333333333447 * sp[78];
    A[7] = -0.03333333333333456 * sp[74] + -0.03333333333333456 * sp[75] + -0.1333333333333331 * sp[76] + -0.03333333333333455 * sp[75] + -0.03333333333333456 * sp[77] + -0.1333333333333331 * sp[78] + -0.03333333333333446 * sp[76] + -0.03333333333333448 * sp[78] + -0.1333333333333331 * sp[79];
    A[8] = -0.03333333333333457 * sp[74] + -0.1333333333333331 * sp[75] + -0.03333333333333455 * sp[76] + -0.03333333333333455 * sp[75] + -0.1333333333333331 * sp[77] + -0.03333333333333451 * sp[78] + -0.0333333333333345 * sp[76] + -0.1333333333333331 * sp[78] + -0.03333333333333447 * sp[79];
    A[9] = -0.133333333333333 * sp[74] + -0.03333333333333444 * sp[75] + -0.03333333333333448 * sp[76] + -0.1333333333333331 * sp[75] + -0.03333333333333451 * sp[77] + -0.03333333333333454 * sp[78] + -0.1333333333333331 * sp[76] + -0.03333333333333442 * sp[78] + -0.03333333333333445 * sp[79];
    A[10] = 0.03333333333333325 * sp[74] + 0.03333333333333331 * sp[75] + 0.0333333333333333 * sp[76];
    A[11] = 0.09999999999999999 * sp[74];
    A[12] = -0.03333333333333337 * sp[75];
    A[13] = -0.03333333333333335 * sp[76];
    A[14] = -0.03333333333333321 * sp[75] + -0.03333333333333317 * sp[76];
    A[15] = -0.03333333333333322 * sp[74] + 0.1000000000000002 * sp[76];
    A[16] = -0.03333333333333324 * sp[74] + 0.1000000000000002 * sp[75];
    A[17] = 0.0333333333333332 * sp[74] + 0.0333333333333332 * sp[75];
    A[18] = 0.03333333333333325 * sp[74] + 0.03333333333333321 * sp[76];
    A[19] = -0.1333333333333336 * sp[74] + -0.1000000000000001 * sp[75] + -0.1000000000000002 * sp[76];
    A[20] = 0.03333333333333339 * sp[75] + 0.03333333333333336 * sp[77] + 0.03333333333333338 * sp[78];
    A[21] = -0.03333333333333337 * sp[75];
    A[22] = 0.09999999999999991 * sp[77];
    A[23] = -0.03333333333333333 * sp[78];
    A[24] = -0.03333333333333342 * sp[77] + 0.09999999999999998 * sp[78];
    A[25] = -0.03333333333333344 * sp[75] + -0.03333333333333351 * sp[78];
    A[26] = 0.09999999999999989 * sp[75] + -0.03333333333333333 * sp[77];
    A[27] = 0.03333333333333342 * sp[75] + 0.03333333333333342 * sp[77];
    A[28] = -0.09999999999999992 * sp[75] + -0.1333333333333336 * sp[77] + -0.09999999999999988 * sp[78];
    A[29] = 0.03333333333333344 * sp[77] + 0.03333333333333351 * sp[78];
    A[30] = 0.03333333333333338 * sp[76] + 0.03333333333333338 * sp[78] + 0.03333333333333335 * sp[79];
    A[31] = -0.03333333333333335 * sp[76];
    A[32] = -0.03333333333333333 * sp[78];
    A[33] = 0.1000000000000002 * sp[79];
    A[34] = 0.09999999999999985 * sp[78] + -0.03333333333333392 * sp[79];
    A[35] = 0.09999999999999991 * sp[76] + -0.0333333333333336 * sp[79];
    A[36] = -0.03333333333333373 * sp[76] + -0.03333333333333385 * sp[78];
    A[37] = -0.09999999999999985 * sp[76] + -0.09999999999999985 * sp[78] + -0.1333333333333338 * sp[79];
    A[38] = 0.03333333333333374 * sp[76] + 0.03333333333333374 * sp[79];
    A[39] = 0.03333333333333369 * sp[78] + 0.0333333333333336 * sp[79];
    A[40] = 0.03333333333333456 * sp[75] + 0.03333333333333455 * sp[77] + 0.03333333333333448 * sp[78] + 0.03333333333333452 * sp[76] + 0.03333333333333449 * sp[78] + 0.03333333333333443 * sp[79];
    A[41] = -0.03333333333333321 * sp[75] + -0.03333333333333317 * sp[76];
    A[42] = -0.03333333333333342 * sp[77] + 0.09999999999999998 * sp[78];
    A[43] = 0.09999999999999985 * sp[78] + -0.03333333333333392 * sp[79];
    A[44] = 0.2666666666666679 * sp[77] + 0.1333333333333341 * sp[78] + 0.1333333333333341 * sp[78] + 0.266666666666668 * sp[79];
    A[45] = 0.2666666666666679 * sp[75] + 0.1333333333333353 * sp[78] + 0.1333333333333341 * sp[76] + 0.1333333333333352 * sp[79];
    A[46] = 0.1333333333333342 * sp[75] + 0.1333333333333348 * sp[77] + 0.2666666666666678 * sp[76] + 0.1333333333333352 * sp[78];
    A[47] = -0.2666666666666678 * sp[75] + -0.2666666666666678 * sp[77] + -0.1333333333333344 * sp[78] + -0.133333333333334 * sp[76] + -0.133333333333334 * sp[78];
    A[48] = -0.1333333333333343 * sp[75] + -0.1333333333333341 * sp[78] + -0.2666666666666679 * sp[76] + -0.1333333333333345 * sp[78] + -0.2666666666666678 * sp[79];
    A[49] = -0.1333333333333342 * sp[77] + -0.1333333333333343 * sp[78] + -0.1333333333333343 * sp[78] + -0.1333333333333342 * sp[79];
    A[50] = 0.03333333333333458 * sp[74] + 0.03333333333333457 * sp[75] + 0.03333333333333448 * sp[76] + 0.03333333333333457 * sp[76] + 0.03333333333333462 * sp[78] + 0.03333333333333455 * sp[79];
    A[51] = -0.03333333333333322 * sp[74] + 0.1000000000000002 * sp[76];
    A[52] = -0.03333333333333344 * sp[75] + -0.03333333333333351 * sp[78];
    A[53] = 0.09999999999999991 * sp[76] + -0.0333333333333336 * sp[79];
    A[54] = 0.2666666666666679 * sp[75] + 0.1333333333333341 * sp[76] + 0.1333333333333353 * sp[78] + 0.1333333333333352 * sp[79];
    A[55] = 0.2666666666666679 * sp[74] + 0.1333333333333353 * sp[76] + 0.1333333333333353 * sp[76] + 0.2666666666666696 * sp[79];
    A[56] = 0.1333333333333342 * sp[74] + 0.1333333333333348 * sp[75] + 0.1333333333333351 * sp[76] + 0.2666666666666695 * sp[78];
    A[57] = -0.2666666666666678 * sp[74] + -0.2666666666666678 * sp[75] + -0.1333333333333345 * sp[76] + -0.1333333333333352 * sp[76] + -0.1333333333333352 * sp[78];
    A[58] = -0.1333333333333342 * sp[74] + -0.1333333333333341 * sp[76] + -0.1333333333333352 * sp[76] + -0.133333333333335 * sp[79];
    A[59] = -0.1333333333333342 * sp[75] + -0.1333333333333342 * sp[76] + -0.1333333333333344 * sp[76] + -0.2666666666666685 * sp[78] + -0.2666666666666686 * sp[79];
    A[60] = 0.03333333333333456 * sp[74] + 0.03333333333333453 * sp[75] + 0.03333333333333448 * sp[76] + 0.0333333333333345 * sp[75] + 0.03333333333333455 * sp[77] + 0.03333333333333447 * sp[78];
    A[61] = -0.03333333333333324 * sp[74] + 0.1000000000000002 * sp[75];
    A[62] = 0.09999999999999989 * sp[75] + -0.03333333333333333 * sp[77];
    A[63] = -0.03333333333333373 * sp[76] + -0.03333333333333385 * sp[78];
    A[64] = 0.1333333333333342 * sp[75] + 0.2666666666666678 * sp[76] + 0.1333333333333348 * sp[77] + 0.1333333333333352 * sp[78];
    A[65] = 0.1333333333333342 * sp[74] + 0.1333333333333351 * sp[76] + 0.1333333333333348 * sp[75] + 0.2666666666666695 * sp[78];
    A[66] = 0.2666666666666677 * sp[74] + 0.1333333333333351 * sp[75] + 0.1333333333333351 * sp[75] + 0.2666666666666693 * sp[77];
    A[67] = -0.1333333333333341 * sp[74] + -0.1333333333333342 * sp[75] + -0.1333333333333347 * sp[75] + -0.1333333333333348 * sp[77];
    A[68] = -0.2666666666666678 * sp[74] + -0.1333333333333344 * sp[75] + -0.2666666666666676 * sp[76] + -0.1333333333333352 * sp[75] + -0.133333333333335 * sp[78];
    A[69] = -0.1333333333333342 * sp[75] + -0.1333333333333341 * sp[76] + -0.1333333333333344 * sp[75] + -0.2666666666666683 * sp[77] + -0.2666666666666683 * sp[78];
    A[70] = -0.03333333333333456 * sp[74] + -0.03333333333333455 * sp[75] + -0.03333333333333446 * sp[76] + -0.03333333333333456 * sp[75] + -0.03333333333333456 * sp[77] + -0.03333333333333448 * sp[78] + -0.1333333333333331 * sp[76] + -0.1333333333333331 * sp[78] + -0.1333333333333331 * sp[79];
    A[71] = 0.0333333333333332 * sp[74] + 0.0333333333333332 * sp[75];
    A[72] = 0.03333333333333342 * sp[75] + 0.03333333333333342 * sp[77];
    A[73] = -0.09999999999999985 * sp[76] + -0.09999999999999985 * sp[78] + -0.1333333333333338 * sp[79];
    A[74] = -0.2666666666666678 * sp[75] + -0.133333333333334 * sp[76] + -0.2666666666666678 * sp[77] + -0.133333333333334 * sp[78] + -0.1333333333333344 * sp[78];
    A[75] = -0.2666666666666678 * sp[74] + -0.1333333333333352 * sp[76] + -0.2666666666666678 * sp[75] + -0.1333333333333352 * sp[78] + -0.1333333333333345 * sp[76];
    A[76] = -0.1333333333333341 * sp[74] + -0.1333333333333347 * sp[75] + -0.1333333333333342 * sp[75] + -0.1333333333333348 * sp[77];
    A[77] = 0.2666666666666677 * sp[74] + 0.2666666666666677 * sp[75] + 0.1333333333333344 * sp[76] + 0.2666666666666677 * sp[75] + 0.2666666666666677 * sp[77] + 0.1333333333333344 * sp[78] + 0.1333333333333344 * sp[76] + 0.1333333333333344 * sp[78] + 0.2666666666666668 * sp[79];
    A[78] = 0.1333333333333342 * sp[74] + 0.133333333333334 * sp[76] + 0.1333333333333342 * sp[75] + 0.1333333333333341 * sp[78] + 0.1333333333333322 * sp[78];
    A[79] = 0.1333333333333341 * sp[75] + 0.1333333333333342 * sp[76] + 0.1333333333333341 * sp[77] + 0.1333333333333342 * sp[78] + 0.1333333333333322 * sp[76];
    A[80] = -0.03333333333333457 * sp[74] + -0.03333333333333455 * sp[75] + -0.0333333333333345 * sp[76] + -0.1333333333333331 * sp[75] + -0.1333333333333331 * sp[77] + -0.1333333333333331 * sp[78] + -0.03333333333333455 * sp[76] + -0.03333333333333451 * sp[78] + -0.03333333333333447 * sp[79];
    A[81] = 0.03333333333333325 * sp[74] + 0.03333333333333321 * sp[76];
    A[82] = -0.09999999999999992 * sp[75] + -0.1333333333333336 * sp[77] + -0.09999999999999988 * sp[78];
    A[83] = 0.03333333333333374 * sp[76] + 0.03333333333333374 * sp[79];
    A[84] = -0.1333333333333343 * sp[75] + -0.2666666666666679 * sp[76] + -0.1333333333333345 * sp[78] + -0.1333333333333341 * sp[78] + -0.2666666666666678 * sp[79];
    A[85] = -0.1333333333333342 * sp[74] + -0.1333333333333352 * sp[76] + -0.1333333333333341 * sp[76] + -0.133333333333335 * sp[79];
    A[86] = -0.2666666666666678 * sp[74] + -0.1333333333333352 * sp[75] + -0.1333333333333344 * sp[75] + -0.2666666666666676 * sp[76] + -0.133333333333335 * sp[78];
    A[87] = 0.1333333333333342 * sp[74] + 0.1333333333333342 * sp[75] + 0.1333333333333322 * sp[78] + 0.133333333333334 * sp[76] + 0.1333333333333341 * sp[78];
    A[88] = 0.2666666666666679 * sp[74] + 0.1333333333333345 * sp[75] + 0.2666666666666677 * sp[76] + 0.1333333333333345 * sp[75] + 0.2666666666666669 * sp[77] + 0.1333333333333344 * sp[78] + 0.2666666666666677 * sp[76] + 0.1333333333333344 * sp[78] + 0.2666666666666675 * sp[79];
    A[89] = 0.1333333333333342 * sp[75] + 0.1333333333333341 * sp[76] + 0.1333333333333322 * sp[75] + 0.1333333333333341 * sp[78] + 0.133333333333334 * sp[79];
    A[90] = -0.133333333333333 * sp[74] + -0.1333333333333331 * sp[75] + -0.1333333333333331 * sp[76] + -0.03333333333333444 * sp[75] + -0.03333333333333451 * sp[77] + -0.03333333333333442 * sp[78] + -0.03333333333333448 * sp[76] + -0.03333333333333454 * sp[78] + -0.03333333333333445 * sp[79];
    A[91] = -0.1333333333333336 * sp[74] + -0.1000000000000001 * sp[75] + -0.1000000000000002 * sp[76];
    A[92] = 0.03333333333333344 * sp[77] + 0.03333333333333351 * sp[78];
    A[93] = 0.03333333333333369 * sp[78] + 0.0333333333333336 * sp[79];
    A[94] = -0.1333333333333342 * sp[77] + -0.1333333333333343 * sp[78] + -0.1333333333333343 * sp[78] + -0.1333333333333342 * sp[79];
    A[95] = -0.1333333333333344 * sp[76] + -0.1333333333333342 * sp[75] + -0.2666666666666685 * sp[78] + -0.1333333333333342 * sp[76] + -0.2666666666666686 * sp[79];
    A[96] = -0.1333333333333344 * sp[75] + -0.1333333333333342 * sp[75] + -0.2666666666666683 * sp[77] + -0.1333333333333341 * sp[76] + -0.2666666666666683 * sp[78];
    A[97] = 0.1333333333333322 * sp[76] + 0.1333333333333341 * sp[75] + 0.1333333333333341 * sp[77] + 0.1333333333333342 * sp[76] + 0.1333333333333342 * sp[78];
    A[98] = 0.1333333333333322 * sp[75] + 0.1333333333333342 * sp[75] + 0.1333333333333341 * sp[78] + 0.1333333333333341 * sp[76] + 0.133333333333334 * sp[79];
    A[99] = 0.2666666666666666 * sp[74] + 0.1333333333333342 * sp[75] + 0.1333333333333342 * sp[76] + 0.1333333333333342 * sp[75] + 0.2666666666666674 * sp[77] + 0.2666666666666674 * sp[78] + 0.1333333333333342 * sp[76] + 0.2666666666666674 * sp[78] + 0.2666666666666674 * sp[79];
}
"""

# C header for Laplace operator tensor tabulation
TABULATE_H = """
void tabulate_tensor_A(double* A, const double* const* w, double* coords, int cell_orientation);
"""


def reference_tensor(element: FiniteElement):
    """Generates code for the Laplace P1(Tetrahedron) reference tensor"""

    element_str = utils.format_filename(str(element))
    filename = f"{element_str}.npy"

    # Generate the reference tensor or load from file
    if os.path.isfile(filename):
        A0 = np.load(filename)
        print("Loaded reference tensor from file.")
    else:
        A0 = generate_ref_tensor(element)
        np.save(filename, A0)
        print("Generated reference tensor.")

    # Eliminate negative zeros
    A0[A0 == 0] = 0

    assert (A0.shape[0] == A0.shape[1])
    assert (A0.shape[2] == A0.shape[3])

    n_dof = A0.shape[0]
    n_dim = A0.shape[2]

    # Flatten reference tensor and convert to CSR sparse matrix
    A0_flat = A0.reshape((n_dof**2, n_dim**2))
    A0_csr = sps.csr_matrix(A0_flat)

    vals = A0_csr.data
    row_ptrs = A0_csr.indptr
    col_idx = A0_csr.indices

    assert(row_ptrs.size == A0_flat.shape[0] + 1)
    assert (row_ptrs[-1] == vals.size)

    # Generate C definitions of sparse arrays
    vals_string = f"alignas(32) static const double A0_vals[{vals.size}] = {{\n#\n}};".replace("#", ", ".join(str(x) for x in A0_csr.data))
    row_ptr_string = f"static const int A0_row_ptr[{row_ptrs.size}] = {{\n#\n}};".replace("#", ", ".join(str(x) for x in A0_csr.indptr))
    col_idx_string = f"static const int A0_col_idx[{col_idx.size}] = {{\n#\n}};".replace("#", ", ".join(str(x) for x in A0_csr.indices))

    # Generate C definition of dense tensor
    A0_string = f"alignas(32) static const double A0[{A0.size}] = {{\n#\n}};"
    numbers = ",\n".join(", ".join(str(x) for x in A0[i,j,:,:].flatten()) for i,j in itertools.product(range(n_dof),range(n_dof)))
    A0_string = A0_string.replace("#", numbers)

    return "\n".join([f"#define N_DOF {n_dof}",
                      f"#define N_DIM {n_dim}",
                      f"#define GT_SIZE {n_dim**2}",
                      f"#define AT_SIZE {n_dof**2}",
                      f"#define A0_NNZ {vals.size}",
                      f"#define VECTORIZED_NNZ {int(np.floor(vals.size / 4)*4)}",
                      vals_string, row_ptr_string, col_idx_string, A0_string]), n_dof, n_dim


class cffi_kernels:
    @staticmethod
    def kernel_manual():
        """Handwritten cell kernel for the Laplace operator"""

        code = """{
            double acc_knl;
            for (int i = 0; i < N_DOF; ++i) {
                for (int j = 0; j < N_DOF; ++j) {
                    acc_knl = 0;
                    for (int k = 0; k < GT_SIZE; ++k) {
                        acc_knl += A0[i*N_DOF*GT_SIZE + j*GT_SIZE + k] * G_T[k];
                    }
                    A_T[i*N_DOF + j] = acc_knl;
                }
            }
        }"""

        return code

    # Following code does not generalize to arbitrary elements
    '''
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
    '''

    @staticmethod
    def kernel_sparse(knl_name: str):
        code = """
    void {knl_name}(double *restrict A_T, 
                         double const *restrict A0_vals,
                         int const *restrict A0_col_idx,
                         int const *restrict A0_row_ptr, 
                         double const *restrict G_T)
    {   
        double acc_A;
        for (int i = 0; i < AT_SIZE; ++i) {
            acc_A = 0;

            for (int j = A0_row_ptr[i]; j < A0_row_ptr[i+1]; ++j) {
                acc_A += A0_vals[j] * G_T[A0_col_idx[j]];
            }

            A_T[i] = acc_A;
        }
    }""".replace("{knl_name}", knl_name)

        header = """void {knl_name}(double *restrict A_T, 
                                    double const *restrict A0_vals,
                                    int const *restrict A0_col_idx,
                                    int const *restrict A0_row_ptr, 
                                    double const *restrict G_T);
        """.replace("{knl_name}", knl_name)

        return code, header

    @staticmethod
    def kernel_sparse_avx(knl_name: str):
        code = """
    #include <immintrin.h>
    void {knl_name}(double *restrict A_T, 
                         double const *restrict A0_vals,
                         int const *restrict A0_col_idx,
                         int const *restrict A0_row_ptr, 
                         double const *restrict G_T)
    {   
        // A0 * G_T multiplication, vectorized
        alignas(32) double A_T_scattered[A0_NNZ];
        for (int i = 0; i < VECTORIZED_NNZ; i+=4) {
            __m256d a0 = _mm256_load_pd(&A0_vals[i]);
            __m256d g = _mm256_set_pd(G_T[A0_col_idx[i+3]], 
                                      G_T[A0_col_idx[i+2]], 
                                      G_T[A0_col_idx[i+1]], 
                                      G_T[A0_col_idx[i]]);
            __m256d res = _mm256_mul_pd(a0, g);
            _mm256_store_pd(&A_T_scattered[i], res);
        }
        
        #if VECTORIZED_NNZ != A0_NNZ
        // A0 * G_T multiplication, remainder
        for (int i = VECTORIZED_NNZ; i < A0_NNZ; ++i) {
            A_T_scattered[i] = A0_vals[i]*G_T[A0_col_idx[i]];
        }
        #endif
        
        // Reduce
        double acc_A;
        for (int i = 0; i < AT_SIZE; ++i) {
            acc_A = 0;
            
            for (int j = A0_row_ptr[i]; j < A0_row_ptr[i+1]; ++j) {
                acc_A += A_T_scattered[j];
            }
            
            A_T[i] = acc_A;
        }
    }""".replace("{knl_name}", knl_name)

        header = """void {knl_name}(double *restrict A_T, 
                                    double const *restrict A0_vals,
                                    int const *restrict A0_col_idx,
                                    int const *restrict A0_row_ptr, 
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

    def useLoopy():
        print("Using Loopy generated kernel.")
        knl_c, knl_h = cffi_kernels.kernel_loopy(knl_name, n_dof, n_dim, verbose)

        knl_call = f"{knl_name}(A_T, &A0[0], &G_T[0]);"
        knl_impl = knl_c + "\n"
        knl_sig = knl_h + "\n"

        return knl_call, knl_impl, knl_sig

    '''
    def useAVX():
        print("Using manual naive AVX kernel.")
        knl_c, knl_h = cffi_kernels.kernel_avx(knl_name)

        # Use hand written kernel
        knl_call = f"{knl_name}(A_T, &A0[0], &G_T[0]);"
        knl_impl = knl_c + "\n"
        knl_sig = knl_h + "\n"

        return knl_call, knl_impl, knl_sig
    '''

    def useSparse():
        print("Using manual kernel on sparse tensor (no AVX).")
        knl_c, knl_h = cffi_kernels.kernel_sparse(knl_name)

        # Use hand written kernel
        knl_call = f"{knl_name}(A_T, &A0_vals[0], &A0_col_idx[0], &A0_row_ptr[0], &G_T[0]);"
        knl_impl = knl_c + "\n"
        knl_sig = knl_h + "\n"

        return knl_call, knl_impl, knl_sig

    def useSparseAvx():
        print("Using manual AVX kernel on sparse tensor.")
        knl_c, knl_h = cffi_kernels.kernel_sparse_avx(knl_name)

        # Use hand written kernel
        knl_call = f"{knl_name}(A_T, &A0_vals[0], &A0_col_idx[0], &A0_row_ptr[0], &G_T[0]);"
        knl_impl = knl_c + "\n"
        knl_sig = knl_h + "\n"

        return knl_call, knl_impl, knl_sig

    knl_call, knl_impl, knl_sig = useSparse()

    if useFFCCode:
        code_c = TABULATE_C_FFC_P2
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
    mesh = generate_mesh(n)
    print("Mesh generated.")

    element = FiniteElement("P", tetrahedron, 2)
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
        print("Finished compiling kernel.")

        # Import the compiled kernel
        kernel_mod = importlib.import_module(f"simd.tmp.{module_name}")
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
        print("Built form.")

    assembler = dolfin.cpp.fem.Assembler([[a]], [L], [bc])
    A = PETScMatrix()
    b = PETScVector()

    # Callable that performs assembly of matrix
    assembly_callable = lambda : assembler.assemble(A, dolfin.cpp.fem.Assembler.BlockType.monolithic)

    # Get timings for assembly of matrix over several runs
    n_runs = 10
    time_avg, time_min, time_max = utils.timing(n_runs, assembly_callable, verbose=True)
    print(f"Timings for element matrix (n={n_runs}) avg: {round(time_avg*1000, 2)}ms, min: {round(time_min*1000, 2)}ms, max: {round(time_max*1000, 2)}ms")

    # Assemble again to get correct results
    A = PETScMatrix()
    assembler.assemble(A, dolfin.cpp.fem.Assembler.BlockType.monolithic)
    assembler.assemble(b, dolfin.cpp.fem.Assembler.BlockType.monolithic)

    Anorm = A.norm(dolfin.cpp.la.Norm.frobenius)
    bnorm = b.norm(dolfin.cpp.la.Norm.l2)
    print(Anorm, bnorm)

    if useCustomKernels:
        # Norms obtained with FFC and n=22
        assert (np.isclose(Anorm, 118.19435458024503))
        #assert (np.isclose(bnorm, 0.009396467472097566))

    return

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