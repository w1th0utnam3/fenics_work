import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmarker.types import TestCase, BenchmarkReport, TestRunArgs, FormTestResult


def parse_args(args):
    parser = argparse.ArgumentParser(prog="{} plot".format(sys.argv[0]),
                                     description="Plots previously generated benchmark reports. Does not require FEniCS.")
    parser.add_argument("report_filename", help="Input filename of the benchmark report that should be plotted")

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
    df = df.rename(index={i: name for i, name in enumerate(form_names)})
    # Create the plot
    df.plot.barh(figsize=(10, len(form_names) * 1.2))

    ax = plt.gca()
    # Fix that horizontal bar plots are in wrong order
    ax.invert_yaxis()
    # Show x axis grid
    ax.grid(axis="x")

    plt.show()

    return
