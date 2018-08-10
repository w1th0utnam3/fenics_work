import ufl
import ffc.compiler
from ffc.fiatinterface import create_element

import sys
import argparse
from copy import copy
from typing import Dict, List, Tuple
import numpy as np

from ttbench.types import TestCase, FormTestData, TestRunParameters
from ttbench.c_code import wrap_tabulate_tensor_code, join_test_wrappers


def parse_args(args):
    parser = argparse.ArgumentParser(prog="{} generate".format(sys.argv[0]),
                                     description="Generates benchmark data. Requires a FEniCS installation.")
    parser.add_argument("data_filename", help="Output filename for the benchmark data that should be generated.")
    return parser.parse_args(args)


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


def eval_with_globals(expr: str, globals_def: str, globals_cache: Dict):
    """
    Evaluates an expression in an environment and returns its result.

    The environments are cached and re-used when the same string is supplied as an environment again.

    :param expr: String containing the expression that will be evaluated with the globals.
    :param globals_def: String defining globals (i.e. an environment) that is required by the expression.
    :return: The return value of the expression.
    """

    namespace = {}
    try:
        namespace = globals_cache[globals_def]
    except KeyError:
        exec(globals_def, namespace)
        globals_cache[globals_def] = namespace

    return eval(expr, copy(namespace))


def compute_shapes(form: ufl.Form):
    """Computes the shapes of the cell tensor, coefficient & coordinate arrays for a form"""

    # Cell tensor
    elements = tuple(arg.ufl_element() for arg in form.arguments())
    fiat_elements = (create_element(element) for element in elements)
    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)
    A_shape = element_dims

    # Coefficient arrays
    ws_shapes = []
    for coefficient in form.coefficients():
        w_element = coefficient.ufl_element()
        w_dim = create_element(w_element).space_dimension()
        ws_shapes.append((w_dim,))
    if len(ws_shapes) > 0:
        max_w_dim = sorted(ws_shapes, key=lambda w_dim: np.prod(w_dim))[-1]
        w_full_shape = (len(ws_shapes), ) + max_w_dim
    else:
        w_full_shape = (0,)

    # Coordinate dofs array
    num_vertices_per_cell = form.ufl_cell().num_vertices()
    gdim = form.ufl_cell().geometric_dimension()
    coords_shape = (num_vertices_per_cell, gdim)

    return A_shape, w_full_shape, coords_shape


def check_shapes(form: ufl.Form, form_def: FormTestData) -> bool:
    """Checks whether shapes from form_def match the actual shapes required by the UFL form"""

    A_shape, w_full_shape, coords_shape = compute_shapes(form)
    if form_def.element_tensor_size != np.prod(A_shape):
        raise RuntimeError("Element tensor size of form '{}' defined in form_data ({}) does not correspond to "
                           "actual size of element tensor ({})!"
                           "".format(form_def.form_name, form_def.element_tensor_size, np.prod(A_shape)))

    if np.prod(w_full_shape) != 0 and np.prod(form_def.coefficients.shape) != np.prod(w_full_shape):
        raise RuntimeError("Coefficient array size of form '{}' defined in form_data ({}->{}) "
                           "does not correspond to actual size of the coefficient array ({}->{})!"
                           "".format(form_def.form_name, form_def.coefficients.shape,
                                     np.prod(form_def.coefficients.shape), w_full_shape, np.prod(w_full_shape)))

    if np.prod(form_def.coord_dofs.shape) != np.prod(coords_shape):
        raise RuntimeError("Coordinate array size of form '{}' defined in form_data ({}->{}) "
                           "does not correspond to actual size of the coordinate array ({}->{})!"
                           "".format(form_def.form_name, form_def.coord_dofs.shape,
                                     np.prod(form_def.coord_dofs.shape), coords_shape, np.prod(coords_shape)))

    return True


def generate_benchmark_code(test_case: TestCase) -> Tuple[Dict[str, List[Tuple[str, int]]], List[Tuple[str, str]]]:
    """
    Generates code for a TestCase.

    :param test_case: The TestCase to process.
    :return: A tuple consisting of:
        - a dict mapping form name -> a list of all test wrapper function names and the index of its implementation
            in the second output value
        - a list containing tuples of all generated C sources and headers
    """

    form_env_cache = dict()
    forms = dict()

    # Generate the UFL forms from their sources (independent of test parameter sets)
    for form_def in test_case.forms:  # type: FormTestData
        form_expr, form_env = form_def.form_code
        form = eval_with_globals(form_expr, form_env, form_env_cache)
        check_shapes(form, form_def)
        forms[form_def.form_name] = form

    # Dict storing generated data (C code, wrapper function names, etc.) for the forms
    test_functions = {form_def.form_name: [] for form_def in test_case.forms}

    # Loop over all run arguments sets (e.g. FFC arguments)
    for j, run_arg_set in enumerate(test_case.run_args):  # type: int, TestRunParameters
        ffc_arg_set = copy(run_arg_set.ffc_args)
        cross_element_width = run_arg_set.cross_element_width
        if cross_element_width > 0:
            ffc_arg_set["cell_batch_size"] = cross_element_width
        else:
            cross_element_width = 1

        # Loop over all forms to generate tabulate_tensor code
        for form_def in test_case.forms:  # type: FormTestData
            form = forms[form_def.form_name]

            # Run FFC (tabulate_tensor code generation)
            raw_function_name, raw_code = compile_form(form, form_def.form_name + "_" + str(j),
                                                       extra_ffc_args=ffc_arg_set)

            gcc_ext_enabled = run_arg_set.ffc_args.get("enable_cross_cell_gcc_ext", False)

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
