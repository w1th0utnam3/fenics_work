import argparse

import benchmarker.execute as execute
import benchmarker.io as io
import benchmarker.form_data as forms
from benchmarker.types import TestCase, TestRunArgs


def gen_test_case() -> TestCase:
    """Generates an example test case."""

    # Macros for ICC
    """
    [("_Float32", "float"),
    ("_Float64","double"),
    ("_Float128", "'long double'"),
    ("_Float32x", "double"),
    ("_Float64x","'long double'")]
    """

    # Default GCC parameters
    gcc_default = {
        "-D__PURE_INTEL_C99_HEADERS__": False,
        "-D_GNU_SOURCE": False,
        "-D_Float32=float -D_Float64='long double' -D_Float32x=double -D_Float64x='long double'": False,
        "-O2": True,
        "-funroll-loops": False,
        "-ftree-vectorize": False,
        "-march=native": True,
        "-mtune=native": False,
    }

    # GCC parameters with auto vectorization
    gcc_auto_vectorize = {
        "-D__PURE_INTEL_C99_HEADERS__": False,
        "-D_GNU_SOURCE": False,
        "-D_Float32=float -D_Float64='long double' -D_Float32x=double -D_Float64x='long double'": False,
        "-O2": True,
        "-funroll-loops": True,
        "-ftree-vectorize": True,
        "-march=native": True,
        "-mtune=native": True,
    }

    # Default FFC parameters
    ffc_default = {"optimize": True}
    # FFC with padding enabled
    ffc_padded = {"optimize": True, "padlen": 4}

    one_element = TestRunArgs(
        name="default",
        cross_element_width=0,
        ffc_args=ffc_default
    )

    one_element_padded = TestRunArgs(
        name="padded",
        cross_element_width=0,
        ffc_args=ffc_padded
    )

    four_elements = TestRunArgs(
        name="4x cross element",
        cross_element_width=4,
        ffc_args=ffc_default
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
            four_elements
        ],
        forms=forms.get_all_forms(),
        reference_case=(0, 0),
        n_repeats=3
    )

    return test_case


def example_generate():
    """Generates example benchmark data and stores it in files."""

    import benchmarker.generate as generate

    test_case = gen_test_case()
    test_fun_names, code_c, code_h = generate.generate_benchmark_code(test_case)
    io.save_generated_test_case_data("example_benchmark_data", test_fun_names, code_c, code_h)


def example_run():
    """Loads example benchmark data and runs it."""

    test_case = gen_test_case()
    test_fun_names, code_c, code_h = io.load_generated_test_case_data("example_benchmark_data")

    report = execute.run_benchmark(test_case, test_fun_names, code_c, code_h)
    io.print_report(test_case, report)


def example_simple():
    """Generates and runs the example benchmark without saving."""

    import benchmarker.generate as generate

    test_case = gen_test_case()

    test_fun_names, code_c, code_h = generate.generate_benchmark_code(test_case)
    report = execute.run_benchmark(test_case, test_fun_names, code_c, code_h)
    io.print_report(test_case, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", help="Generates benchmark data. Requires a FEniCS installtion.", action="store_true")
    parser.add_argument("--run", help="Runs previously generated benchmark data. Does not require FEniCS.", action="store_true")
    args = parser.parse_args()

    if args.generate or args.run:
        if args.generate:
            example_generate()
        if args.run:
            example_run()
    else:
        example_simple()

    return 0
