import ufl
import ffc.compiler

from copy import copy
from typing import Dict, List, Tuple

from benchmarker.types import TestCase, FormTestData, TestRunArgs
from benchmarker.c_code import wrap_tabulate_tensor_code, join_test_wrappers


def compile_form(form: ufl.Form, prefix: str, extra_ffc_args=None) -> Tuple[str, str]:
    """Compiles an UFL form and returns the source code of the tabulate_tensor function."""

    prefix = prefix.lower()
    parameters = ffc.parameters.default_parameters()

    # Merge extra args into parameter dict
    if extra_ffc_args is not None:
        for key, value in extra_ffc_args.items():
            parameters[key] = value

    # For now, don't support lists of forms
    assert not isinstance(form, list)
    form_index = 0

    # Call FFC
    code_h, code_c = ffc.compiler.compile_form(form, prefix=prefix, parameters=parameters)

    # Build the concrete C function name of the generated function
    function_name = "tabulate_tensor_{}_cell_integral_{}_otherwise".format(prefix, form_index)
    # Find section of generated code that contains the tabulate_tensor function
    index_start = code_c.index("void {}(".format(function_name))
    index_end = code_c.index("ufc_cell_integral* create_{}_cell_integral_{}_otherwise(void)".format(prefix, form_index),
                             index_start)
    # Extract tabulate_tensor definition
    tabulate_tensor_code = code_c[index_start:index_end].strip()

    return function_name, tabulate_tensor_code


def generate_benchmark_code(test_case: TestCase) -> Tuple[Dict[str, List[Tuple[str, int]]], List[Tuple[str, str]]]:
    """
    Generates code for a TestCase.

    :param test_case: The TestCase to process.
    :return: A tuple consisting of:
        - a dict mapping form name -> a list of all test wrapper function names and the index of its implementation
            in the second output value
        - a list containing tuples of all generated C sources and headers
    """

    test_functions = {form_def.form_name: [] for form_def in test_case.forms}

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
            form = form_def.form_gen()

            # Run FFC (tabulate_tensor code generation)
            raw_function_name, raw_code = compile_form(form, form_def.form_name + "_" + str(j),
                                                       extra_ffc_args=ffc_arg_set)

            gcc_ext_enabled = run_arg_set.ffc_args.get("enable_cross_element_gcc_ext", False)

            # Wrap tabulate_tensor code in test runner function
            wrapper_function_name = "_test_runner_" + raw_function_name
            code, signature = wrap_tabulate_tensor_code(wrapper_function_name, raw_function_name, raw_code, form_def,
                                                        cross_element_width, gcc_ext_enabled)

            # Store generated content
            test_functions[form_def.form_name].append((wrapper_function_name, code, signature))

    # Generate separate test files for every form and ffc parameter combination
    def separate_file_per_test():
        test_fun_names = {form_def.form_name: [] for form_def in test_case.forms}
        test_funs = []

        for form_name, test_function_list in test_functions.items():
            for wrapper_function_name, code, signature in test_function_list:
                code_index = len(test_funs)
                test_fun_names[form_name].append((wrapper_function_name, code_index))

                code_c, code_h = join_test_wrappers([code], [signature])
                test_funs.append((code_c, code_h))

        return test_fun_names, test_funs

    # Concatenate all test codes
    def join_all_test():
        test_fun_names = {form_def.form_name: [] for form_def in test_case.forms}
        codes = []
        signatures = []

        for form_name, test_function_list in test_functions.items():
            for wrapper_function_name, code, signature in test_function_list:
                test_fun_names[form_name].append((wrapper_function_name, 0))
                codes.append(code)
                signatures.append(signature)

        code_c, code_h = join_test_wrappers(codes, signatures)

        return test_fun_names, [(code_c, code_h)]

    return separate_file_per_test()
