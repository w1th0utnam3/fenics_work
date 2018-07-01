import os
import json
import pickle
from typing import Dict, List, Tuple

import numpy as np

from benchmarker.types import TestCase, BenchmarkReport, TestRunArgs, FormTestResult


def print_report(test_case: TestCase, report: BenchmarkReport):
    """Prints a report generated by execute_test."""

    print("")
    print("Benchmark report")
    print("-" * 20)
    print("Total runtime: {:.2f}s".format(report.total_runtime))

    # Get the length of the longest run args name
    longest_name = 0
    for run_arg_set in test_case.run_args:  # type: TestRunArgs
        longest_name = np.maximum(longest_name, len(run_arg_set.name))

    # Loop over results of all forms
    for form_idx, (form_name, form_results) in enumerate(report.results.items()):
        print("{}".format(form_name))
        print("Results for form '{}', test run with n={} elements".format(form_name, test_case.forms[form_idx].n_elems))

        for i, compiler_arg_set in enumerate(test_case.compiler_args):
            active_compile_args = [arg for arg, use in compiler_arg_set.items() if use]
            print("\tCompiled with flags '{}'".format(', '.join(active_compile_args)))

            for j, run_arg_set in enumerate(test_case.run_args):
                result = form_results[(i, j)]  # type: FormTestResult

                print(
                    "\t\t{:<{name_length}} | avg: {:>8.2f}ms | min: {:>8.2f}ms | max: {:>8.2f}ms | speedup: {:>5.2f}x | result: {:>20.14e} | result ok: {}".format(
                        run_arg_set.name,
                        result.avg * 1000,
                        result.min * 1000,
                        result.max * 1000,
                        result.speedup,
                        result.result_val,
                        result.result_ok,
                        name_length=longest_name + 1))

            print("")

        print("-" * 40)
        print("")


def save_generated_data(name: str, test_fun_names: Dict, codes: List[Tuple[str, str]], path: str = ""):
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

    for i, (code_c, code_h) in enumerate(codes):
        store_string(output_basename + "_code_{}.c".format(i), code_c)
        store_string(output_basename + "_code_{}.h".format(i), code_h)


def load_generated_data(name: str, path: str = "") -> Tuple[Dict, List[Tuple[str, str]]]:
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

    n_code_files = 0
    test_fun_json = load_string(input_basename + "_funs.json")
    test_fun_names = json.loads(test_fun_json)

    # Convert lists from json file to tuples
    for form_name, test_function_list in test_fun_names.items():
        for i in range(len(test_function_list)):
            fun_name, code_idx = test_function_list[i][0], test_function_list[i][1]
            test_function_list[i] = (fun_name, code_idx)

            n_code_files = max(n_code_files, code_idx)

    codes = []
    for i in range(n_code_files + 1):
        code_c = load_string(input_basename + "_code_{}.c".format(i))
        code_h = load_string(input_basename + "_code_{}.h".format(i))
        codes.append((code_c, code_h))

    return test_fun_names, codes


def save_report(filename: str, report: BenchmarkReport):
    """Stores a benchmark report to the specified file."""

    with open(filename, mode="wb") as f:
        pickle.dump(report, f)


def load_report(filename: str) -> BenchmarkReport:
    """Loads a benchmark report from the specified file."""

    with open(filename, mode="rb") as f:
        report = pickle.load(f)

    return report
