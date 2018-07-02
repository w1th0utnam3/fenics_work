import benchmarker.form_data as form_data
from benchmarker.types import TestCase, TestRunParameters, BenchmarkReport

from typing import List

def gen_test_case() -> TestCase:
    """Generates an example test case."""

    # Warning: every parameter needs its own entry due to os.exec behavior
    # Default GCC parameters
    gcc_default = {
        "-O2": True,
        "-funroll-loops": False,
        "-ftree-vectorize": False,
        "-march=skylake": True,
        "-mtune=skylake": False,
    }

    # GCC parameters with auto vectorization
    gcc_auto_vectorize = {
        "-O2": True,
        "-funroll-loops": True,
        "-ftree-vectorize": True,
        "-march=skylake": True,
        "-mtune=skylake": True,
    }

    # Default FFC parameters
    ffc_default = {"optimize": True}
    # FFC with padding enabled
    ffc_padded = {"optimize": True, "padlen": 4}

    one_element = TestRunParameters(
        name="ffc default",
        cross_element_width=0,
        ffc_args=ffc_default
    )

    one_element_padded = TestRunParameters(
        name="ffc padded",
        cross_element_width=0,
        ffc_args=ffc_padded
    )

    four_elements = TestRunParameters(
        name="4x cross element",
        cross_element_width=4,
        ffc_args=ffc_default
    )

    four_elements_gcc_exts = TestRunParameters(
        name="4x ce, gcc exts",
        cross_element_width=4,
        ffc_args={"optimize": True, "enable_cross_element_gcc_ext": True}
    )

    four_elements_to_scalars = TestRunParameters(
        name="4x ce, scalar sp",
        cross_element_width=4,
        ffc_args={"optimize": True, "enable_cross_element_array_conv": True}
    )

    four_elements_to_scalars_fused = TestRunParameters(
        name="4x ce, ssp, fused",
        cross_element_width=4,
        ffc_args={"optimize": True, "enable_cross_element_array_conv": True, "enable_cross_element_fuse": True}
    )

    # Combine all options: benchmark will be cartesian product of compiler_args, run_args and forms
    test_case = TestCase(
        compiler_args=[
            gcc_default,
            gcc_auto_vectorize
        ],
        run_args=[
            one_element,
            one_element_padded,
            four_elements,
            four_elements_gcc_exts,
            #four_elements_to_scalars,
            #four_elements_to_scalars_fused
        ],
        forms=form_data.get_all_forms(),
        #forms=[form_data.laplace_p2tet_coefficient_p1tet()],
        reference_case=(0, 0),
        n_repeats=3
    )

    # Optional defines when using ICC
    test_case.compiler_defines = [
        ("__PURE_INTEL_C99_HEADERS__", ""),
        ("_Float32", "float"),
        ("_Float64", "double"),
        ("_Float128", "long double"),
        ("_Float32x", "double"),
        ("_Float64x", "long double")
    ]

    return test_case


def plot_params_single_file(test_case: TestCase, report: BenchmarkReport):
    compile_args = ["gcc default", "gcc + vec flags"]

    combinations_to_plot = [
        (0, 1),
        (0, 2),
    ]

    reference_ind = (0, 0)

    return compile_args, combinations_to_plot, reference_ind


def plot_params_multiple_files(test_cases: List[TestCase], reports: List[BenchmarkReport]):
    compile_args_gcc = ["gcc", "gcc + vec flags"]
    compile_args_icc = ["icc", "icc + vec flags"]

    combinations_gcc = [
        (1, 1),
        (1, 2),
        (1, 3),
    ]

    combinations_icc = [
        (1, 0),
        (1, 1),
        (1, 2),
    ]

    compile_args = [compile_args_gcc, compile_args_icc]
    combinations_to_plot = [combinations_gcc, combinations_icc]
    reference_ind = (0, 0, 0)

    return compile_args, combinations_to_plot, reference_ind
