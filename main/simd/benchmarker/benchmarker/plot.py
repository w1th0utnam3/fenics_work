import sys
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmarker.types import TestCase, BenchmarkReport


def parse_args(args):
    parser = argparse.ArgumentParser(prog="{} plot".format(sys.argv[0]),
                                     description="Plots previously generated benchmark reports. Does not require FEniCS.")
    parser.add_argument("report_filename", nargs="+",
                        help="Input filename of the benchmark report that should be plotted.")

    return parser.parse_args(args)


def plot_report(test_case: TestCase, report: BenchmarkReport,
                compiler_arg_names: List[str],
                visualize_combinations: List[Tuple[int, int]],
                reference_combination: Tuple[int, int]):
    """
    Plots a benchmark report.

    :param test_case: The TestCase that was used to produce the benchmark.
    :param report: The BenchmarkReport that should be plotted.
    :param compiler_arg_names: Names that should be displayed for the sets of compiler parameters.
    :param visualize_combinations: List of tuples with indices that indicate which compiler and runtime parameter pairs
        should be plotted
    :param reference: The reference case that should be used to calculate speedup.
    """

    form_names = [str(name) for name in report.results.keys()]
    run_args = [run_args.name for run_args in test_case.run_args]

    # Collect columns (visualized by one bar per form)
    columns = ["{}, {}".format(compiler_arg_names[i], run_args[j]) for (i, j) in visualize_combinations]

    speedup = np.zeros((len(report.results), len(columns)))

    # Calculate and collect speedup values
    for form_idx, (form_name, form_results) in enumerate(report.results.items()):
        reference_time = form_results[reference_combination].avg

        for k, (i, j) in enumerate(visualize_combinations):
            avg_time = form_results[(i, j)].avg
            speedup[form_idx, k] = reference_time / avg_time

    df = pd.DataFrame(speedup, columns=columns)
    # Assign form names to rows
    labels = [form_name + " ({:.2f}ms)".format(form_results[reference_combination].avg * 1000) for form_name, form_results in report.results.items()]
    df = df.rename(index={i: label for i, label in enumerate(labels)})
    # Create the plot
    df.plot.barh(figsize=(10, len(form_names) * 1.4))

    ax = plt.gca()

    # Fix that horizontal bar plots are in wrong order
    ax.invert_yaxis()

    ax.grid(axis="x")
    ax.set_xlabel("Speedup factor in comparison to gcc and ffc with default parameters")

    plt.show()
    return


def join_over_compile_args(test_cases: List[TestCase],
                           reports: List[BenchmarkReport],
                           compiler_arg_names: List[List[str]],
                           visualize_combinations: List[List[Tuple[int, int]]],
                           reference_combination: Tuple[int, int, int]):
    # Check that all high level lists have same length
    assert len(test_cases) == len(reports) == len(compiler_arg_names) == len(visualize_combinations)
    # Check that all test cases have the same forms
    form_names = [[form.form_name for form in case.forms] for case in test_cases]
    assert form_names.count(form_names[0]) == len(form_names)

    joined_compile_arg_names = []
    joined_compile_args = []
    joined_run_args = []

    compile_arg_offsets = [0]
    run_arg_offsets = [0]

    # Compute offsets of all compile and run parameter sets
    for i, case in enumerate(test_cases):
        compile_arg_offsets.append(compile_arg_offsets[i - 1] + len(case.compiler_args))
        joined_compile_args += case.compiler_args
        joined_compile_arg_names += compiler_arg_names[i]

        run_arg_offsets.append(run_arg_offsets[i - 1] + len(case.run_args))
        joined_run_args += case.run_args

    # Make sure that there is a name for every compile parameter set
    assert len(joined_compile_arg_names) == len(joined_compile_args)

    # Apply offset to the reference combination
    ref_case = reference_combination[0]
    ref_compile_args = reference_combination[1] + compile_arg_offsets[ref_case]
    ref_run_args = reference_combination[2] + run_arg_offsets[ref_case]
    joined_reference_combination = ref_compile_args, ref_run_args

    # Apply offset to all visualize combinations
    joined_visualize_combinations = []
    for case_idx, case_visualize_combinations in enumerate(visualize_combinations):
        for i, j in case_visualize_combinations:
            joined_visualize_combinations.append((i + compile_arg_offsets[case_idx],
                                                  j + run_arg_offsets[case_idx]))

    # Join the test cases
    joined_test_case = TestCase(compiler_args=joined_compile_args,
                                run_args=joined_run_args,
                                forms=test_cases[0].forms,
                                reference_case=joined_reference_combination,
                                n_repeats=test_cases[0].n_repeats)

    # Join all benchmark results per form
    joined_results = {form_name: dict() for form_name in form_names[0]}
    for case_idx, report in enumerate(reports):
        for form_name, form_results in report.results.items():
            for (i, j), result in form_results.items():
                offset_tuple = (i + compile_arg_offsets[case_idx],
                                j + run_arg_offsets[case_idx])
                joined_results[form_name][offset_tuple] = result

    # Generate joined report
    joined_report = BenchmarkReport(sum([report.total_runtime for report in reports]),
                                    joined_results)

    return (joined_test_case, joined_report, joined_compile_arg_names,
            joined_visualize_combinations, joined_reference_combination)
