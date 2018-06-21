#include <stdalign.h>

#define N_ELEMS 4
#define A_SIZE 100

void tabulate_tensor_elem(
	double* restrict A, 
	const double* const* w,
	const double* restrict coordinate_dofs,
	int cell_orientation)
{
    // Quadrature rules
    alignas(32) static const double weights5[5] = { -0.1333333333333333, 0.075, 0.075, 0.075, 0.075 };
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [entities][points][dofs]
    // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
    // PM* dimensions: [entities][dofs][dofs]
    alignas(32) static const double FE15_C0_D001_Q5[1][5][7] =
        { { { 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0 },
            { 0.3333333333333341, -0.3333333333333373, 0.6666666666666681, 2.000000000000007, 0.0, -0.6666666666666669, -2.0 },
            { 0.333333333333334, -0.3333333333333375, 2.000000000000003, 0.6666666666666723, 0.0, -2.000000000000001, -0.666666666666666 },
            { 0.3333333333333339, 1.0, 0.6666666666666653, 0.6666666666666736, -1.333333333333334, -0.6666666666666664, -0.6666666666666671 },
            { -1.0, -0.3333333333333377, 0.6666666666666683, 0.6666666666666723, 1.333333333333334, -0.6666666666666672, -0.6666666666666669 } } };
    alignas(32) static const double FE15_C0_D010_Q5[1][5][7] =
        { { { 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, -1.0 },
            { 0.333333333333335, -0.3333333333333356, 0.6666666666666675, 2.000000000000006, -0.6666666666666672, 0.0, -2.000000000000001 },
            { 0.3333333333333346, 1.0, 0.6666666666666675, 0.6666666666666732, -0.6666666666666672, -1.333333333333334, -0.6666666666666672 },
            { 0.3333333333333347, -0.3333333333333351, 2.000000000000002, 0.6666666666666701, -2.000000000000002, 0.0, -0.6666666666666667 },
            { -1.0, -0.3333333333333359, 0.6666666666666675, 0.6666666666666715, -0.6666666666666672, 1.333333333333333, -0.6666666666666672 } } };
    alignas(32) static const double FE15_C0_D100_Q5[1][5][7] =
        { { { 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0 },
            { 0.3333333333333343, 1.0, 0.666666666666667, 0.6666666666666672, -0.666666666666667, -0.6666666666666675, -1.333333333333333 },
            { 0.3333333333333348, -0.3333333333333346, 0.6666666666666671, 2.000000000000001, -0.6666666666666671, -2.000000000000002, 0.0 },
            { 0.3333333333333347, -0.3333333333333345, 2.000000000000002, 0.666666666666667, -2.000000000000002, -0.6666666666666674, 0.0 },
            { -1.0, -0.333333333333334, 0.6666666666666672, 0.6666666666666672, -0.6666666666666672, -0.6666666666666675, 1.333333333333333 } } };
    alignas(32) static const double FE8_C0_Q5[1][5][4] =
        { { { 0.2500000000000001, 0.25, 0.25, 0.25 },
            { 0.1666666666666668, 0.5, 0.1666666666666667, 0.1666666666666666 },
            { 0.1666666666666668, 0.1666666666666666, 0.5, 0.1666666666666666 },
            { 0.1666666666666668, 0.1666666666666666, 0.1666666666666667, 0.5 },
            { 0.5, 0.1666666666666666, 0.1666666666666667, 0.1666666666666666 } } };
    alignas(32) static const double FE9_C0_D001_Q5[1][1][2] = { { { -1.0, 1.0 } } };
    // Unstructured piecewise computations
    alignas(32) double J_c4[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c4[i_simd] = coordinate_dofs[i_simd + 4] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 7] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double J_c8[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c8[i_simd] = coordinate_dofs[i_simd + 4 * 2] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 11] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double J_c5[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c5[i_simd] = coordinate_dofs[i_simd + 4] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 10] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double J_c7[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c7[i_simd] = coordinate_dofs[i_simd + 4 * 2] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 8] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double J_c0[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c0[i_simd] = coordinate_dofs[i_simd] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 3] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double J_c1[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c1[i_simd] = coordinate_dofs[i_simd] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 6] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double J_c6[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c6[i_simd] = coordinate_dofs[i_simd + 4 * 2] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 5] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double J_c3[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c3[i_simd] = coordinate_dofs[i_simd + 4] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 4] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double J_c2[4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        J_c2[i_simd] = coordinate_dofs[i_simd] * FE9_C0_D001_Q5[0][0][0] + coordinate_dofs[i_simd + 4 * 9] * FE9_C0_D001_Q5[0][0][1];
    alignas(32) double sp[74][4];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[0][i_simd] = J_c4[i_simd] * J_c8[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[1][i_simd] = J_c5[i_simd] * J_c7[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[2][i_simd] = sp[0][i_simd] + -1 * sp[1][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[3][i_simd] = J_c0[i_simd] * sp[2][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[4][i_simd] = J_c5[i_simd] * J_c6[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[5][i_simd] = J_c3[i_simd] * J_c8[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[6][i_simd] = sp[4][i_simd] + -1 * sp[5][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[7][i_simd] = J_c1[i_simd] * sp[6][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[8][i_simd] = sp[3][i_simd] + sp[7][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[9][i_simd] = J_c3[i_simd] * J_c7[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[10][i_simd] = J_c4[i_simd] * J_c6[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[11][i_simd] = sp[9][i_simd] + -1 * sp[10][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[12][i_simd] = J_c2[i_simd] * sp[11][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[13][i_simd] = sp[8][i_simd] + sp[12][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[14][i_simd] = sp[2][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[15][i_simd] = J_c3[i_simd] * (-1 * J_c8[i_simd]);
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[16][i_simd] = sp[4][i_simd] + sp[15][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[17][i_simd] = sp[16][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[18][i_simd] = sp[11][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[19][i_simd] = sp[14][i_simd] * sp[14][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[20][i_simd] = sp[14][i_simd] * sp[17][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[21][i_simd] = sp[18][i_simd] * sp[14][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[22][i_simd] = sp[17][i_simd] * sp[17][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[23][i_simd] = sp[18][i_simd] * sp[17][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[24][i_simd] = sp[18][i_simd] * sp[18][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[25][i_simd] = J_c2[i_simd] * J_c7[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[26][i_simd] = J_c8[i_simd] * (-1 * J_c1[i_simd]);
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[27][i_simd] = sp[25][i_simd] + sp[26][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[28][i_simd] = sp[27][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[29][i_simd] = J_c0[i_simd] * J_c8[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[30][i_simd] = J_c6[i_simd] * (-1 * J_c2[i_simd]);
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[31][i_simd] = sp[29][i_simd] + sp[30][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[32][i_simd] = sp[31][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[33][i_simd] = J_c1[i_simd] * J_c6[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[34][i_simd] = J_c0[i_simd] * J_c7[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[35][i_simd] = sp[33][i_simd] + -1 * sp[34][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[36][i_simd] = sp[35][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[37][i_simd] = sp[28][i_simd] * sp[28][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[38][i_simd] = sp[28][i_simd] * sp[32][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[39][i_simd] = sp[28][i_simd] * sp[36][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[40][i_simd] = sp[32][i_simd] * sp[32][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[41][i_simd] = sp[32][i_simd] * sp[36][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[42][i_simd] = sp[36][i_simd] * sp[36][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[43][i_simd] = sp[37][i_simd] + sp[19][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[44][i_simd] = sp[38][i_simd] + sp[20][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[45][i_simd] = sp[39][i_simd] + sp[21][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[46][i_simd] = sp[40][i_simd] + sp[22][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[47][i_simd] = sp[41][i_simd] + sp[23][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[48][i_simd] = sp[24][i_simd] + sp[42][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[49][i_simd] = J_c1[i_simd] * J_c5[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[50][i_simd] = J_c2[i_simd] * J_c4[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[51][i_simd] = sp[49][i_simd] + -1 * sp[50][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[52][i_simd] = sp[51][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[53][i_simd] = J_c2[i_simd] * J_c3[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[54][i_simd] = J_c0[i_simd] * J_c5[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[55][i_simd] = sp[53][i_simd] + -1 * sp[54][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[56][i_simd] = sp[55][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[57][i_simd] = J_c0[i_simd] * J_c4[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[58][i_simd] = J_c1[i_simd] * J_c3[i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[59][i_simd] = sp[57][i_simd] + -1 * sp[58][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[60][i_simd] = sp[59][i_simd] / sp[13][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[61][i_simd] = sp[52][i_simd] * sp[52][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[62][i_simd] = sp[52][i_simd] * sp[56][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[63][i_simd] = sp[60][i_simd] * sp[52][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[64][i_simd] = sp[56][i_simd] * sp[56][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[65][i_simd] = sp[60][i_simd] * sp[56][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[66][i_simd] = sp[60][i_simd] * sp[60][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[67][i_simd] = sp[43][i_simd] + sp[61][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[68][i_simd] = sp[44][i_simd] + sp[62][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[69][i_simd] = sp[45][i_simd] + sp[63][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[70][i_simd] = sp[46][i_simd] + sp[64][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[71][i_simd] = sp[47][i_simd] + sp[65][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[72][i_simd] = sp[48][i_simd] + sp[66][i_simd];
    for (int i_simd = 0; i_simd < 4; ++i_simd)
        sp[73][i_simd] = fabs(sp[13][i_simd]);
    // UFLACS block mode: full
    alignas(32) double BF0[7][7][4] = { 0 };
    // UFLACS block mode: full
    alignas(32) double BF1[7][7][4] = { 0 };
    // UFLACS block mode: full
    alignas(32) double BF2[7][7][4] = { 0 };
    // UFLACS block mode: full
    alignas(32) double BF3[7][7][4] = { 0 };
    // UFLACS block mode: full
    alignas(32) double BF4[7][7][4] = { 0 };
    // UFLACS block mode: full
    alignas(32) double BF5[7][7][4] = { 0 };
    // UFLACS block mode: full
    alignas(32) double BF6[7][7][4] = { 0 };
    // UFLACS block mode: full
    alignas(32) double BF7[7][7][4] = { 0 };
    // UFLACS block mode: full
    alignas(32) double BF8[7][7][4] = { 0 };
    for (int iq = 0; iq < 5; ++iq)
    {
        // Quadrature loop body setup (num_points=5)
        // Unstructured varying computations for num_points=5
        alignas(32) double w0[4];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            w0[i_simd] = 0.0;
        for (int ic = 0; ic < 4; ++ic)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                w0[i_simd] += w[0][i_simd + 4 * ic] * FE8_C0_Q5[0][iq][ic];
        alignas(32) double sv5[12][4];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[0][i_simd] = sp[67][i_simd] * w0[i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[1][i_simd] = sp[68][i_simd] * w0[i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[2][i_simd] = sp[69][i_simd] * w0[i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[3][i_simd] = sp[70][i_simd] * w0[i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[4][i_simd] = sp[71][i_simd] * w0[i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[5][i_simd] = sp[72][i_simd] * w0[i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[6][i_simd] = sv5[0][i_simd] * sp[73][i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[7][i_simd] = sv5[1][i_simd] * sp[73][i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[8][i_simd] = sv5[2][i_simd] * sp[73][i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[9][i_simd] = sv5[3][i_simd] * sp[73][i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[10][i_simd] = sv5[4][i_simd] * sp[73][i_simd];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            sv5[11][i_simd] = sv5[5][i_simd] * sp[73][i_simd];
        // UFLACS block mode: full
        alignas(32) double fw0[4];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            fw0[i_simd] = sv5[6][i_simd] * weights5[iq];
        alignas(32) double TF0[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF0[i][i_simd] = fw0[i_simd] * FE15_C0_D100_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF0[i][j][i_simd] += TF0[i][i_simd] * FE15_C0_D100_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double fw1[4];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            fw1[i_simd] = sv5[7][i_simd] * weights5[iq];
        alignas(32) double TF1[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF1[i][i_simd] = fw1[i_simd] * FE15_C0_D100_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF1[i][j][i_simd] += TF1[i][i_simd] * FE15_C0_D010_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double fw2[4];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            fw2[i_simd] = sv5[8][i_simd] * weights5[iq];
        alignas(32) double TF2[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF2[i][i_simd] = fw2[i_simd] * FE15_C0_D100_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF2[i][j][i_simd] += TF2[i][i_simd] * FE15_C0_D001_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double TF3[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF3[i][i_simd] = fw1[i_simd] * FE15_C0_D010_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF3[i][j][i_simd] += TF3[i][i_simd] * FE15_C0_D100_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double fw3[4];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            fw3[i_simd] = sv5[9][i_simd] * weights5[iq];
        alignas(32) double TF4[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF4[i][i_simd] = fw3[i_simd] * FE15_C0_D010_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF4[i][j][i_simd] += TF4[i][i_simd] * FE15_C0_D010_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double fw4[4];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            fw4[i_simd] = sv5[10][i_simd] * weights5[iq];
        alignas(32) double TF5[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF5[i][i_simd] = fw4[i_simd] * FE15_C0_D010_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF5[i][j][i_simd] += TF5[i][i_simd] * FE15_C0_D001_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double TF6[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF6[i][i_simd] = fw2[i_simd] * FE15_C0_D001_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF6[i][j][i_simd] += TF6[i][i_simd] * FE15_C0_D100_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double TF7[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF7[i][i_simd] = fw4[i_simd] * FE15_C0_D001_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF7[i][j][i_simd] += TF7[i][i_simd] * FE15_C0_D010_Q5[0][iq][j];
        // UFLACS block mode: full
        alignas(32) double fw5[4];
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            fw5[i_simd] = sv5[11][i_simd] * weights5[iq];
        alignas(32) double TF8[7][4];
        for (int i = 0; i < 7; ++i)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                TF8[i][i_simd] = fw5[i_simd] * FE15_C0_D001_Q5[0][iq][i];
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int i_simd = 0; i_simd < 4; ++i_simd)
                    BF8[i][j][i_simd] += TF8[i][i_simd] * FE15_C0_D001_Q5[0][iq][j];
    }
    // UFLACS block mode: preintegrated
    for (int k = 0; k < 100; ++k)
        for (int i_simd = 0; i_simd < 4; ++i_simd)
            A[i_simd + 4 * k] = 0.0;
    static const int DM0[7] = { 0, 1, 5, 6, 7, 8, 9 };
    static const int DM1[7] = { 0, 2, 4, 6, 7, 8, 9 };
    static const int DM2[7] = { 0, 3, 4, 5, 7, 8, 9 };
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM0[i] + DM0[j])] += BF0[i][j][i_simd];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM0[i] + DM1[j])] += BF1[i][j][i_simd];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM0[i] + DM2[j])] += BF2[i][j][i_simd];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM1[i] + DM0[j])] += BF3[i][j][i_simd];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM1[i] + DM1[j])] += BF4[i][j][i_simd];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM1[i] + DM2[j])] += BF5[i][j][i_simd];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM2[i] + DM0[j])] += BF6[i][j][i_simd];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM2[i] + DM1[j])] += BF7[i][j][i_simd];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int i_simd = 0; i_simd < 4; ++i_simd)
                A[i_simd + 4 * (10 * DM2[i] + DM2[j])] += BF8[i][j][i_simd];
}

#undef N_ELEMS
