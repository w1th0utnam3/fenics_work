#include <math.h>

#ifndef _MSC_VER
    #include <stdalign.h>
    typedef double double4 __attribute__ ((vector_size (32)));
#else
    #define alignas(X)
    #define restrict __restrict
#endif

typedef double ufc_scalar_t;

// BEGIN CODE FOR tabulate_tensor_laplace_p2tet_coefficient_p1tet_0_cell_integral_0_otherwise

void tabulate_tensor_laplace_p2tet_coefficient_p1tet_0_cell_integral_0_otherwise(double4* restrict A, const double4* const* w,
                                    const double4* restrict coordinate_dofs,
                                    int cell_orientation)
{
    typedef double double4 __attribute__ ((vector_size (32)));
    double4* restrict A_x = (double4* restrict)A;
    const double4* const* w_x = (const double4* const*)w;
    const double4* restrict coordinate_dofs_x = (const double4* restrict)coordinate_dofs;
    // Quadrature rules
    alignas(32) static const double weights5[5] = { -0.1333333333333333, 0.075, 0.075, 0.075, 0.075 };
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [entities][points][dofs]
    // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
    // PM* dimensions: [entities][dofs][dofs]
    alignas(32) static const double FE15_C0_D001_Q5[1][5][7] =
        { { { 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0 },
            { 0.3333333333333342, -0.3333333333333372, 0.666666666666668, 2.000000000000008, 0.0, -0.6666666666666665, -2.0 },
            { 0.3333333333333339, -0.3333333333333374, 2.000000000000002, 0.6666666666666718, 0.0, -2.0, -0.6666666666666663 },
            { 0.333333333333334, 1.0, 0.6666666666666652, 0.6666666666666737, -1.333333333333334, -0.666666666666666, -0.6666666666666671 },
            { -1.0, -0.3333333333333374, 0.6666666666666682, 0.6666666666666721, 1.333333333333334, -0.6666666666666669, -0.6666666666666667 } } };
    alignas(32) static const double FE15_C0_D010_Q5[1][5][7] =
        { { { 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, -1.0 },
            { 0.3333333333333348, -0.3333333333333359, 0.6666666666666674, 2.000000000000006, -0.6666666666666673, 0.0, -2.000000000000001 },
            { 0.3333333333333344, 1.0, 0.6666666666666674, 0.6666666666666731, -0.6666666666666671, -1.333333333333334, -0.6666666666666673 },
            { 0.3333333333333347, -0.3333333333333356, 2.000000000000002, 0.6666666666666701, -2.000000000000002, 0.0, -0.6666666666666667 },
            { -1.0, -0.3333333333333363, 0.6666666666666674, 0.6666666666666716, -0.6666666666666673, 1.333333333333333, -0.6666666666666671 } } };
    alignas(32) static const double FE15_C0_D100_Q5[1][5][7] =
        { { { 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0 },
            { 0.3333333333333342, 1.0, 0.6666666666666671, 0.6666666666666671, -0.6666666666666669, -0.6666666666666674, -1.333333333333333 },
            { 0.3333333333333345, -0.333333333333335, 0.6666666666666672, 2.000000000000001, -0.6666666666666671, -2.000000000000002, 0.0 },
            { 0.3333333333333345, -0.3333333333333345, 2.000000000000002, 0.6666666666666667, -2.000000000000002, -0.666666666666667, 0.0 },
            { -1.0, -0.333333333333334, 0.6666666666666673, 0.6666666666666671, -0.6666666666666671, -0.6666666666666674, 1.333333333333333 } } };
    alignas(32) static const double FE8_C0_Q5[1][5][4] =
        { { { 0.2500000000000001, 0.25, 0.25, 0.25 },
            { 0.1666666666666668, 0.5, 0.1666666666666667, 0.1666666666666667 },
            { 0.1666666666666668, 0.1666666666666666, 0.5, 0.1666666666666667 },
            { 0.1666666666666668, 0.1666666666666666, 0.1666666666666667, 0.5 },
            { 0.5, 0.1666666666666665, 0.1666666666666667, 0.1666666666666667 } } };
    alignas(32) static const double FE9_C0_D001_Q5[1][1][2] = { { { -1.0, 1.0 } } };
    // Unstructured piecewise computations
    const double4 J_c4 = coordinate_dofs_x[1] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[7] * FE9_C0_D001_Q5[0][0][1];
    const double4 J_c8 = coordinate_dofs_x[2] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[11] * FE9_C0_D001_Q5[0][0][1];
    const double4 J_c5 = coordinate_dofs_x[1] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[10] * FE9_C0_D001_Q5[0][0][1];
    const double4 J_c7 = coordinate_dofs_x[2] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[8] * FE9_C0_D001_Q5[0][0][1];
    const double4 J_c0 = coordinate_dofs_x[0] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[3] * FE9_C0_D001_Q5[0][0][1];
    const double4 J_c1 = coordinate_dofs_x[0] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[6] * FE9_C0_D001_Q5[0][0][1];
    const double4 J_c6 = coordinate_dofs_x[2] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[5] * FE9_C0_D001_Q5[0][0][1];
    const double4 J_c3 = coordinate_dofs_x[1] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[4] * FE9_C0_D001_Q5[0][0][1];
    const double4 J_c2 = coordinate_dofs_x[0] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs_x[9] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double4 sp[74];
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
    for (int i_elem = 0; i_elem < 4; ++i_elem)
        sp[73][i_elem] = fabs(sp[13][i_elem]);
    // UFLACS block mode: full
    alignas(32) double4 BF0[7][7] = {{ 0 }};
    // UFLACS block mode: full
    alignas(32) double4 BF1[7][7] = {{ 0 }};
    // UFLACS block mode: full
    alignas(32) double4 BF2[7][7] = {{ 0 }};
    // UFLACS block mode: full
    alignas(32) double4 BF3[7][7] = {{ 0 }};
    // UFLACS block mode: full
    alignas(32) double4 BF4[7][7] = {{ 0 }};
    // UFLACS block mode: full
    alignas(32) double4 BF5[7][7] = {{ 0 }};
    // UFLACS block mode: full
    alignas(32) double4 BF6[7][7] = {{ 0 }};
    // UFLACS block mode: full
    alignas(32) double4 BF7[7][7] = {{ 0 }};
    // UFLACS block mode: full
    alignas(32) double4 BF8[7][7] = {{ 0 }};
    for (int iq = 0; iq < 5; ++iq)
    {
        // Quadrature loop body setup (num_points=5)
        // Unstructured varying computations for num_points=5
        double4 w0;
        for (int i_elem = 0; i_elem < 4; ++i_elem)
            w0[i_elem] = 0.0;
        for (int ic = 0; ic < 4; ++ic)
            w0 += w_x[0][ic] * FE8_C0_Q5[0][iq][ic];
        alignas(32) double4 sv5[12];
        sv5[0] = sp[67] * w0;
        sv5[1] = sp[68] * w0;
        sv5[2] = sp[69] * w0;
        sv5[3] = sp[70] * w0;
        sv5[4] = sp[71] * w0;
        sv5[5] = sp[72] * w0;
        sv5[6] = sv5[0] * sp[73];
        sv5[7] = sv5[1] * sp[73];
        sv5[8] = sv5[2] * sp[73];
        sv5[9] = sv5[3] * sp[73];
        sv5[10] = sv5[4] * sp[73];
        sv5[11] = sv5[5] * sp[73];
        // UFLACS block mode: full
        const double4 fw0 = sv5[6] * weights5[iq];
        alignas(32) double4 TF0[7];
        for (int i = 0; i < 7; ++i)
            TF0[i] = fw0 * FE15_C0_D100_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF0[i][j] += TF0[i] * FE15_C0_D100_Q5[0][iq][j];
        // UFLACS block mode: full
        const double4 fw1 = sv5[7] * weights5[iq];
        alignas(32) double4 TF1[7];
        for (int i = 0; i < 7; ++i)
            TF1[i] = fw1 * FE15_C0_D100_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF1[i][j] += TF1[i] * FE15_C0_D010_Q5[0][iq][j];
        // UFLACS block mode: full
        const double4 fw2 = sv5[8] * weights5[iq];
        alignas(32) double4 TF2[7];
        for (int i = 0; i < 7; ++i)
            TF2[i] = fw2 * FE15_C0_D100_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF2[i][j] += TF2[i] * FE15_C0_D001_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double4 TF3[7];
        for (int i = 0; i < 7; ++i)
            TF3[i] = fw1 * FE15_C0_D010_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF3[i][j] += TF3[i] * FE15_C0_D100_Q5[0][iq][j];
        // UFLACS block mode: full
        const double4 fw3 = sv5[9] * weights5[iq];
        alignas(32) double4 TF4[7];
        for (int i = 0; i < 7; ++i)
            TF4[i] = fw3 * FE15_C0_D010_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF4[i][j] += TF4[i] * FE15_C0_D010_Q5[0][iq][j];
        // UFLACS block mode: full
        const double4 fw4 = sv5[10] * weights5[iq];
        alignas(32) double4 TF5[7];
        for (int i = 0; i < 7; ++i)
            TF5[i] = fw4 * FE15_C0_D010_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF5[i][j] += TF5[i] * FE15_C0_D001_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double4 TF6[7];
        for (int i = 0; i < 7; ++i)
            TF6[i] = fw2 * FE15_C0_D001_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF6[i][j] += TF6[i] * FE15_C0_D100_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double4 TF7[7];
        for (int i = 0; i < 7; ++i)
            TF7[i] = fw4 * FE15_C0_D001_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF7[i][j] += TF7[i] * FE15_C0_D010_Q5[0][iq][j];
        // UFLACS block mode: full
        const double4 fw5 = sv5[11] * weights5[iq];
        alignas(32) double4 TF8[7];
        for (int i = 0; i < 7; ++i)
            TF8[i] = fw5 * FE15_C0_D001_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                BF8[i][j] += TF8[i] * FE15_C0_D001_Q5[0][iq][j];
    }
    // UFLACS block mode: preintegrated unroll
    for (int k = 0; k < 100; ++k)
        for (int i_elem = 0; i_elem < 4; ++i_elem)
            A_x[k][i_elem] = 0.0;
    // Tensor copyout
    static const int DM0[7] = { 0, 1, 5, 6, 7, 8, 9 };
    static const int DM1[7] = { 0, 2, 4, 6, 7, 8, 9 };
    static const int DM2[7] = { 0, 3, 4, 5, 7, 8, 9 };
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM0[i] + DM0[j]] += BF0[i][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM0[i] + DM1[j]] += BF1[i][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM0[i] + DM2[j]] += BF2[i][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM1[i] + DM0[j]] += BF3[i][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM1[i] + DM1[j]] += BF4[i][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM1[i] + DM2[j]] += BF5[i][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM2[i] + DM0[j]] += BF6[i][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM2[i] + DM1[j]] += BF7[i][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            A_x[10 * DM2[i] + DM2[j]] += BF8[i][j];
}

#define A_SIZE 400
#define A_SIZE_EXPANDED (400 % 4 == 0) ? 400 : 400 - (400 % 4) + 4
#define W_DIM1_SIZE 1
#define W_DIM2_SIZE 16
#define DOF_DIM1_SIZE 4
#define DOF_DIM2_SIZE 12

double _test_runner_tabulate_tensor_laplace_p2tet_coefficient_p1tet_0_cell_integral_0_otherwise(
    int n,
    ufc_scalar_t* w_vals,
    double* coord_vals)
{
    // Copy values into aligned storage
    alignas(32) double w[W_DIM1_SIZE][W_DIM2_SIZE];
    for (int i = 0; i < W_DIM1_SIZE; ++i)
        for (int j = 0; j < W_DIM2_SIZE; ++j)
            w[i][j] = w_vals[i*W_DIM2_SIZE + j];

    double* w_ptrs[W_DIM1_SIZE];
    for (int i = 0; i < W_DIM1_SIZE; ++i)
        w_ptrs[i] = &w[i][0];

    alignas(32) double coords[DOF_DIM1_SIZE][DOF_DIM2_SIZE];
    for (int i = 0; i < DOF_DIM1_SIZE; ++i)
        for (int j = 0; j < DOF_DIM2_SIZE; ++j)
            coords[i][j] = coord_vals[i*DOF_DIM2_SIZE + j];

    // Allocate element tensor space once
    alignas(32) double A[A_SIZE_EXPANDED] = { 0.0 };

    double acc[4] = { 0.0 };
    for(int i = 0; i < n; ++i) {
        tabulate_tensor_laplace_p2tet_coefficient_p1tet_0_cell_integral_0_otherwise((double4*)&A[0], (const double4* const*)&w_ptrs[0], (const double4*)&coords[0][0], 0);
        
        // Reduce element tensor to use output
        for(int j = 0; j < A_SIZE; j+=4) {
            acc[0 + 0] += fabs(A[j + 0]);
            acc[0 + 1] += fabs(A[j + 1]);
            acc[0 + 2] += fabs(A[j + 2]);
            acc[0 + 3] += fabs(A[j + 3]);
        }

        // Increment coordinates to have varying inputs
        for(int j = 0; j < DOF_DIM1_SIZE; ++j)
            for(int k = 0; k < DOF_DIM2_SIZE; ++k)
                coords[j][k] += 0.01;
    }

    return acc[0] + acc[1] + acc[2] + acc[3];
}

// END CODE FOR tabulate_tensor_laplace_p2tet_coefficient_p1tet_0_cell_integral_0_otherwise

#include <stdio.h>

int main()
{
    double w_vals[W_DIM1_SIZE*W_DIM2_SIZE] = {0};
    double coord_vals[DOF_DIM1_SIZE*DOF_DIM2_SIZE] = {0};

    double val = 0.0;
    for (int i = 0; i < W_DIM1_SIZE; ++i) {
        for (int j = 0; j < W_DIM2_SIZE; ++j) {
            w_vals[i*W_DIM2_SIZE + j] = val;
            val += 0.1;
        }
    }

    val = 0.0;
    for (int i = 0; i < DOF_DIM1_SIZE; ++i) {
        for (int j = 0; j < DOF_DIM2_SIZE; ++j) {
            coord_vals[i*DOF_DIM2_SIZE + j] = val;
            val += 0.1;
        }
    }

    double res = _test_runner_tabulate_tensor_laplace_p2tet_coefficient_p1tet_0_cell_integral_0_otherwise((200*200*200)/4, w_vals, coord_vals);
    printf("Rounding:\t%f\n", res);
}
