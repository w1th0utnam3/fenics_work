import argparse

import benchmarker.execute as execute
import benchmarker.io as io
import benchmarker.form_data as forms
from benchmarker.types import TestCase, TestRunArgs


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

    four_elements_0 = TestRunArgs(
        name="4x ce, gcc exts",
        cross_element_width=4,
        ffc_args={"optimize": True, "enable_cross_element_gcc_ext": True}
    )

    four_elements_1 = TestRunArgs(
        name="4x ce, scalar sp",
        cross_element_width=4,
        ffc_args={"optimize": True, "enable_cross_element_array_conv": True}
    )

    four_elements_2 = TestRunArgs(
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
            four_elements_0
        ],
        forms=forms.get_all_forms(),
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


def example_generate():
    """Generates example benchmark data and stores it in files."""

    import benchmarker.generate as generate

    test_case = gen_test_case()
    test_fun_names, codes = generate.generate_benchmark_code(test_case)
    io.save_generated_data("example_benchmark_data", test_fun_names, codes)


def example_run(report_filename: str):
    """Loads example benchmark data and runs it."""

    test_case = gen_test_case()
    test_fun_names, codes = io.load_generated_data("example_benchmark_data")

    report = execute.run_benchmark(test_case, test_fun_names, codes)

    io.save_report(report_filename, report)
    io.print_report(test_case, report)


def example_plot(report_filename: str):
    """Loads example benchmark output and plots it."""

    import benchmarker.plot as plot

    test_case = gen_test_case()
    report = io.load_report(report_filename)

    compile_args = ["default", "-ftree-vectorize"]

    combinations_to_plot = [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (1, 3)
    ]

    reference_ind = (0,0)

    plot.plot_report(test_case, report, compile_args, combinations_to_plot, reference_ind)


def example_simple():
    """Generates and runs the example benchmark without saving."""

    import benchmarker.generate as generate

    test_case = gen_test_case()

    test_fun_names, codes = generate.generate_benchmark_code(test_case)
    report = execute.run_benchmark(test_case, test_fun_names, codes)
    io.print_report(test_case, report)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser("generate", add_help=False)
    subparsers.add_parser("run", add_help=False)
    subparsers.add_parser("plot", add_help=False)

    args, unknown_args = parser.parse_known_args()

    if args.command == "generate":
        from benchmarker.generate import parse_args
        args = parse_args(unknown_args)

        example_generate()

    elif args.command == "run":
        from benchmarker.execute import parse_args
        args = parse_args(unknown_args)

        example_run(args.report_filename)

    elif args.command == "plot":
        from benchmarker.plot import parse_args
        args = parse_args(unknown_args)

        example_plot(args.report_filename)

    else:
        example_simple()

    return 0
