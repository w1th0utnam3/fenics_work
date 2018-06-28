from typing import Tuple

from benchmarker.types import FormTestData

import simd.utils as utils

TEST_RUN_SKELETON = """
#include <math.h>
#include <stdalign.h>

typedef double ufc_scalar_t;
typedef double double4 __attribute__ ((vector_size (32)));

{tabulate_tensor_wrappers}
"""


TEST_RUNNER_CODE = """// BEGIN CODE FOR {tabulate_tensor_fun_name}

{tabulate_tensor_fun_code}

#define A_SIZE {A_SIZE_val}
#define W_DIM1_SIZE {W_DIM1_SIZE_val}
#define W_DIM2_SIZE {W_DIM2_SIZE_val}
#define DOF_DIM1_SIZE {DOF_DIM1_SIZE_val}
#define DOF_DIM2_SIZE {DOF_DIM2_SIZE_val}

double {test_runner_fun_name}(
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

    // Allocate element tensor space once (should be set to zero inside of tabulate_tensor)
    alignas(32) double A[A_SIZE];

    double result = 0.0;
    for(int i = 0; i < n; ++i) {
        {tabulate_tensor_fun_name}(({scalar_type}*)&A[0], (const {scalar_type}* const*)&w_ptrs[0], (const {scalar_type}*)&coords[0][0], 0);

        // Reduce element tensor to use output
        for(int j = 0; j < A_SIZE; ++j) {
            result += fabs(A[j]);
        }

        // Increment coordinates to have varying inputs
        for(int j = 0; j < DOF_DIM1_SIZE; ++j)
            for(int k = 0; k < DOF_DIM2_SIZE; ++k)
                coords[j][k] += 0.01;
    }

    return result;
}

#undef A_SIZE
#undef W_DIM1_SIZE
#undef W_DIM2_SIZE
#undef DOF_DIM1_SIZE
#undef DOF_DIM2_SIZE

// END CODE FOR {tabulate_tensor_fun_name}
"""


TEST_RUNNER_SIGNATURE = "double {test_runner_fun_name}(" \
                        "int n, " \
                        "double* w_vals, " \
                        "double* coord_vals);"


def wrap_tabulate_tensor_code(test_name: str,
                              function_name: str,
                              form_code: str,
                              form_def: FormTestData,
                              cross_element_width: int = 1,
                              gcc_ext_enabled: bool = False) -> Tuple[str, str]:

    scalar_type = "double"
    if gcc_ext_enabled:
        replacement_args = [
            ("ufc_scalar_t* restrict A", "double4* restrict A"),
            ("const ufc_scalar_t* const* w", "const double4* const* w"),
            ("const double* restrict coordinate_dofs", "const double4* restrict coordinate_dofs"),
        ]

        form_code = utils.replace_strings(form_code, replacement_args)
        scalar_type = "double4"

    code_strs = [
        ("{A_SIZE_val}", str(form_def.element_tensor_size * cross_element_width)),
        ("{W_DIM1_SIZE_val}", str(form_def.coefficients.shape[0])),
        ("{W_DIM2_SIZE_val}", str(form_def.coefficients.shape[1] * cross_element_width)),
        ("{DOF_DIM1_SIZE_val}", str(form_def.coord_dofs.shape[0])),
        ("{DOF_DIM2_SIZE_val}", str(form_def.coord_dofs.shape[1] * cross_element_width)),
        ("{test_runner_fun_name}", test_name),
        ("{tabulate_tensor_fun_name}", function_name),
        ("{tabulate_tensor_fun_code}", form_code),
        ("{scalar_type}", scalar_type)
    ]

    wrapped_code = utils.replace_strings(TEST_RUNNER_CODE, code_strs)
    signature = TEST_RUNNER_SIGNATURE.replace("{test_runner_fun_name}", test_name)

    return wrapped_code, signature


def join_test_wrappers(test_codes, test_signatures) -> Tuple[str, str]:
    full_c = TEST_RUN_SKELETON.replace("{tabulate_tensor_wrappers}", "\n".join(test_codes))
    full_h = "\n".join(test_signatures)
    return full_c, full_h
