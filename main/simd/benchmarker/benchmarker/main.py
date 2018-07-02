import argparse

import benchmarker.execute as execute
import benchmarker.config as config
import benchmarker.io as io
import benchmarker.form_data as forms
from benchmarker.types import TestCase, TestRunParameters


# TODO: remove speedup calculation before plotting?
# TODO: add name for compile parameter set during test case definition
# TODO: proper configuration of plotting


def example_generate(data_filename: str):
    """Generates example benchmark data and stores it in files."""
    import benchmarker.generate as generate

    test_case = config.gen_test_case()
    test_fun_names, codes = generate.generate_benchmark_code(test_case)
    io.save_generated_data(data_filename, test_case, test_fun_names, codes)


def example_run(data_filename: str, report_filename: str):
    """Loads example benchmark data and runs it."""

    test_case, test_fun_names, codes = io.load_generated_data(data_filename)

    report = execute.run_benchmark(test_case, test_fun_names, codes)

    io.save_report(report_filename, test_case, report)
    io.print_report(test_case, report)


def example_plot(report_filenames: str):
    """Loads example benchmark output and plots it."""
    import benchmarker.plot as plot

    if len(report_filenames) == 1:
        test_case, report = io.load_report(report_filenames[0])

        io.print_report(test_case, report)
        plot.plot_report(test_case, report, *config.plot_params_single_file())
    else:
        cases_and_reports = [io.load_report(report_filename) for report_filename in report_filenames]

        # Print all reports
        for test_case, report in cases_and_reports:
            io.print_report(test_case, report)

        test_cases = [case for case, report in cases_and_reports]
        reports = [report for case, report in cases_and_reports]

        joined_data = plot.join_over_compile_args(test_cases, reports, *config.plot_params_multiple_files())
        plot.plot_report(*joined_data)


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

        example_generate(args.data_filename)

    elif args.command == "run":
        from benchmarker.execute import parse_args
        args = parse_args(unknown_args)

        example_run(args.data_filename, args.report_filename)

    elif args.command == "plot":
        from benchmarker.plot import parse_args
        args = parse_args(unknown_args)

        example_plot(args.report_filename)

    else:
        parser.print_help()

    return 0
