import os
import json

import numpy as np
from copy import copy
from typing import Dict, Tuple

from benchmarker.types import FormTestData, FormTestResult, TestCase, TestRunArgs, BenchmarkReport
from benchmarker.c_code import wrap_tabulate_tensor_code, join_test_wrappers
from benchmarker.compilation import compile_form, compile_cffi, import_cffi
import benchmarker.forms as forms

import simd.utils as utils


def gen_test_case() -> TestCase:
    """Generates an example test case."""

    # Default GCC parameters
    gcc_default = {
        "-O2": True,
        "-funroll-loops": False,
        "-ftree-vectorize": False,
        "-march=native": True,
        "-mtune=native": False,
    }

    # GCC parameters with auto vectorization
    gcc_auto_vectorize = {
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


def generate_benchmark_code(test_case: TestCase) -> Tuple[Dict, str, str]:
    """
    Generates code for a TestCase.

    :param test_case: The TestCase to process.
    :return: A tuple consisting of:
        - a dict containing the C function names of the individual tabulate_tensor wrapper functions
        - a str containing the C source code of all tabulate_tensor functions and their wrappers
        - a str containing the C header for all test wrappers
    """

    # Dict that will be used for the function names of all test wrappers
    test_fun_names = {form_def.form_name: [] for form_def in test_case.forms}

    # List of individual function implementations
    test_codes = []
    # List of individial function headers
    test_signatures = []

    # Loop over all run arguments sets (e.g. FFC arguments)
    for j, run_arg_set in enumerate(test_case.run_args):  # type: int, TestRunArgs
        ffc_arg_set = copy(run_arg_set.ffc_args)
        cross_element_width = run_arg_set.cross_element_width
        if cross_element_width > 0:
            ffc_arg_set["cross_element_width"] = cross_element_width
        else:
            cross_element_width = 1

        # Loop over all forms to generate tabulate_tensor code
        for form_def in test_case.forms:  # type: FormTestData
            # Generate the UFL form
            form = form_def.code_gen()

            # Run FFC (tabulate_tensor code generation)
            raw_function_name, raw_code = compile_form(form, form_def.form_name + "_" + str(j),
                                                       extra_ffc_args=ffc_arg_set)
            # Wrap tabulate_tensor code in test runner function
            test_name = "_test_runner_" + raw_function_name
            code, signature = wrap_tabulate_tensor_code(test_name, raw_function_name, raw_code, form_def,
                                                        cross_element_width)

            # Store generated content
            test_fun_names[form_def.form_name].append(test_name)
            test_codes.append(code)
            test_signatures.append(signature)

    # Concatenate all test codes
    code_c, code_h = join_test_wrappers(test_codes, test_signatures)

    return test_fun_names, code_c, code_h


def run_benchmark(test_case: TestCase, test_fun_names: Dict, code_c: str, code_h: str) -> BenchmarkReport:
    """
    Runs a benchmark TestCase and generates a corresponding BenchmarkReport.

    :param test_case: The TestCase to process.
    :param test_fun_names: A dict containing the C function names of the test wrappers, from generate_benchmark_code.
    :param code_c: The C code of the test case.
    :param code_h: The C header of the test case.
    :return: Benchmark report: dict over all form names, containing dicts (i,j) -> FormTestResult, where (i,j) are
        combinations of compiler parameters and form compiler parameters.
    """

    raw_results = {form_def.form_name: dict() for form_def in test_case.forms}

    # Step 1: Run benchmark
    # ---

    # Outermost loop over all compiler argument sets
    for i, compiler_arg_set in enumerate(test_case.compiler_args):
        active_compile_args = [arg for arg, use in compiler_arg_set.items() if use]

        # Step 1a: Code compilation
        # ---

        # Build the test module
        compile_cffi("_benchmark_{}".format(i), code_c, code_h, compiler_args=active_compile_args, verbose=True)
        ffi, lib = import_cffi("_benchmark_{}".format(i))

        # Step 1b: Run benchmark
        # ---

        # Loop over the forms
        for form_idx, (form_name, form_fun_names) in enumerate(test_fun_names.items()):
            # Loop over the function names of each test case
            for j, fun_name in enumerate(form_fun_names):
                run_arg_set = test_case.run_args[j]  # type: TestRunArgs
                form_data = test_case.forms[form_idx]  # type: FormTestData

                n_elem = form_data.n_elems
                w = form_data.coefficients
                coords_dof = form_data.coord_dofs

                # If cross element vectorization is enabled, tile coefficient and dof arrays accordingly
                cross_element_width = run_arg_set.cross_element_width
                if cross_element_width > 0:
                    n_elem = int(n_elem / cross_element_width)

                    w = np.tile(np.expand_dims(w, 2), cross_element_width)
                    coords_dof = np.tile(np.expand_dims(coords_dof, 2), cross_element_width)

                # Get pointers to numpy arrays
                w_ptr = ffi.cast("double*", w.ctypes.data)
                coords_dof_ptr = ffi.cast("double*", coords_dof.ctypes.data)

                # Get the function that should be called
                fun = getattr(lib, fun_name)

                # How many times the test should be run
                n_runs = test_case.n_repeats
                # The actual test callable
                test_callable = lambda: fun(n_elem, w_ptr, coords_dof_ptr)

                # Run the timing
                avg, min, max = utils.timing(n_runs, test_callable, verbose=True, name=fun_name)

                # Store result
                raw_results[form_name][(i, j)] = avg, min, max

    # Step 2: Generate report and calculate speedup)
    # ---

    # Dict for results that will be returned (includes speedup)
    results = {form_def.form_name: dict() for form_def in test_case.forms}

    # Loop over results of all forms
    for form_idx, (form_name, form_results) in enumerate(raw_results.items()):
        # Obtain the reference time for the form
        reference_time = form_results[(test_case.reference_case)][0]

        for i, compiler_arg_set in enumerate(test_case.compiler_args):
            for j, run_arg_set in enumerate(test_case.run_args):
                raw_result = form_results[(i, j)]
                speedup = reference_time / raw_result[0]

                # Store the result values including speedup
                results[form_name][(i, j)] = FormTestResult(*raw_result, speedup)

    return results


def print_report(test_case: TestCase, report: BenchmarkReport):
    """Prints a report generated by execute_test."""

    print("")
    print("Benchmark report")
    print("-" * 20)

    # Get the length of the longest run args name
    longest_name = 0
    for run_arg_set in test_case.run_args:  # type: TestRunArgs
        longest_name = np.maximum(longest_name, len(run_arg_set.name))

    # Loop over results of all forms
    for form_idx, (form_name, form_results) in enumerate(report.items()):
        print("{}".format(form_name))
        print("Results for form '{}', test run with n={} elements".format(form_name, test_case.forms[form_idx].n_elems))

        for i, compiler_arg_set in enumerate(test_case.compiler_args):
            active_compile_args = [arg for arg, use in compiler_arg_set.items() if use]
            print("\tCompiled with flags '{}'".format(', '.join(active_compile_args)))

            for j, run_arg_set in enumerate(test_case.run_args):
                result = form_results[(i, j)]  # type: FormTestResult

                print(
                    "\t\t{:<{name_length}} | avg: {:>8.2f}ms | min: {:>8.2f}ms | max: {:>8.2f}ms | speedup: {:>5.2f}x".format(
                        run_arg_set.name,
                        result.avg * 1000,
                        result.min * 1000,
                        result.max * 1000,
                        result.speedup,
                        name_length=longest_name + 1))

            print("")

        print("-" * 40)
        print("")


def save_generated_test_case_data(name: str, test_fun_names: Dict, code_c: str, code_h: str, path: str = ""):
    """
    Stores the specified generated test case data in a set of files.

    :param name: The base name for the set files.
    :param path: Optional path where the files should be stored (otherwise uses working directory).
    """

    output_basename = os.path.join(path, name)

    def store_string(filename: str, content: str):
        with open(filename, mode="w") as f:
            f.write(content)

    test_fun_json = json.dumps(test_fun_names, indent=4)
    store_string(output_basename + "_funs.json", test_fun_json)
    store_string(output_basename + "_code.c", code_c)
    store_string(output_basename + "_code.h", code_h)


def load_generated_test_case_data(name: str, path: str = "") ->  Tuple[Dict, str, str]:
    """
    Loads a set of generated test case data from files.

    :param name: The base name for the set of files as given to the save function.
    :param path: Optional path where the file are stored (otherwise uses working directory).
    :return: Tuple that can be used as input to run a benchmark.
    """

    input_basename = os.path.join(path, name)

    def load_string(filename: str) -> str:
        with open(filename, mode="r") as f:
            content = f.read()
            return content

    test_fun_json = load_string(input_basename + "_funs.json")
    code_c = load_string(input_basename + "_code.c")
    code_h = load_string(input_basename + "_code.h")

    test_fun_names = json.loads(test_fun_json)
    return test_fun_names, code_c, code_h


def run():
    test_case = gen_test_case()

    test_fun_names, code_c, code_h = generate_benchmark_code(test_case)

    save_generated_test_case_data("some_test", test_fun_names, code_c, code_h)

    del test_fun_names
    del code_c
    del code_h

    test_fun_names, code_c, code_h = load_generated_test_case_data("some_test")

    report = run_benchmark(test_case, test_fun_names, code_c, code_h)
    print_report(test_case, report)

    return
