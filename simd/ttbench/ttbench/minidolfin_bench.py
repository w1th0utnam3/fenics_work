import ufl
import timeit
import numpy

from petsc4py import PETSc

from minidolfin.meshing import build_unit_square_mesh, build_unit_cube_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import build_sparsity_pattern
from minidolfin.dofmap import pattern_to_csr
from minidolfin.petsc import create_matrix_from_csr
from minidolfin.assembling import assemble, assemble_vectorized

import ttbench.form_data as form_data
import ttbench.io as io
from ttbench.generate import eval_with_globals
from ttbench.types import TestCase, TestRunParameters, BenchmarkReport, FormTestResult

def run_minidolfin_benchmark():
    form_data_list = form_data.get_all_forms()

    form_env_cache = dict()
    forms = dict()

    mesh_tri = build_unit_square_mesh(100, 100)
    mesh_tet = build_unit_cube_mesh(20, 20, 20)

    dofmaps = {}

    benched_forms = []
    results = {}

    # Generate the UFL forms from their sources (independent of test parameter sets)
    for form_def in form_data_list:
        form_expr, form_env = form_def.form_code
        forms[form_def.form_name] = eval_with_globals(form_expr, form_env, form_env_cache)

    for form_def in form_data_list:
        form_name = form_def.form_name
        print("\n\n{}".format(form_name))

        # Get the form
        a = forms[form_name]    # type: ufl.Form

        # Check coefficients
        if len(a.coefficients()) != 0:
            print("Form {} has {} coefficient(s), minidolfin does not support coefficients. "
                   "Skipping".format(form_name, len(a.coefficients())))
            continue

        # Extract form arguments
        arguments = a.arguments()
        if len(arguments) != 2:
            print("Form {} has {} argument(s), minidolfin only supports 2! "
                   "Skipping.".format(form_name, len(arguments)))
            continue

        # Extract used elements
        elements = {arg.ufl_element() for arg in arguments}

        if len(elements) != 1:
            print("Form {} has {} element types. No support for more than 1 element type. Elements: {}, "
                  "Skipping.".format(form_name, len(elements), elements))
            continue

        element, = elements

        # Get the right mesh
        if element.cell() == ufl.tetrahedron:
            mesh = mesh_tet
        elif element.cell() == ufl.triangle:
            mesh = mesh_tri
        else:
            print("Form {} uses '{}' cells. Only support for tet and tri cells! "
                  "Skipping.".format(form_name, element.cell))
            continue

        tdim = mesh.reference_cell.get_dimension()
        print('Number cells: {}'.format(mesh.num_entities(tdim)))

        # Build dofmap
        if element in dofmaps:
            dofmap = dofmaps[element]
        else:
            dofmap = build_dofmap(element, mesh)
            dofmaps[element] = dofmap

        print('Number dofs: {}'.format(dofmap.dim))

        def assemble_test(assemble_fun):
            # Build sparsity pattern and create matrix
            pattern = build_sparsity_pattern(dofmap)
            i, j = pattern_to_csr(pattern)
            A = create_matrix_from_csr((i, j))

            # Run and time assembly
            t_ass = -timeit.default_timer()
            assemble_fun(A, dofmap, a, form_compiler="ffc")
            t_ass += timeit.default_timer()
            print('Assembly time a: {}'.format(t_ass))

            norm = A.norm(PETSc.NormType.NORM_FROBENIUS)

            return t_ass, norm

        print("Default assemble...")
        t_ass_ref, norm_ref = assemble_test(assemble)
        print("Vectorized assemble...")
        t_ass_vec, norm_vec = assemble_test(assemble_vectorized)
        error = numpy.abs(norm_ref - norm_vec)

        print("")
        print("norm_ref: {:.3e}, norm_vec: {:.3e}, err: {:.3e}".format(norm_ref,
                                                                       norm_vec,
                                                                       error))
        print("assembly t_ref: {:.2f}ms, t_vec: {:.2f}ms, speedup: {:.3f}x".format(t_ass_ref * 1000,
                                                                                   t_ass_vec * 1000,
                                                                                   t_ass_ref / t_ass_vec))

        result_ok = numpy.allclose(norm_ref, norm_vec, rtol=1e-7, atol=1e-10)
        if not result_ok:
            print("Significant error of matrix norms: {}!".format(error))

        # Add form data to list of benchmarked forms
        form_def.n_elems = mesh.num_entities(tdim)
        benched_forms.append(form_def)
        # Store test results
        results[form_name] = {
            (0,0): FormTestResult(t_ass_ref, t_ass_ref, t_ass_ref, 1.0, norm_ref, "Reference"),
            (0,1): FormTestResult(t_ass_vec, t_ass_vec, t_ass_vec, t_ass_ref / t_ass_vec, norm_vec, result_ok)
        }


    # Generate fake test case
    test_case = TestCase([{}], [TestRunParameters("default assembly", -1, {}),
                                TestRunParameters("simd assembly", -1, {})],
                         benched_forms, (0,0), 1)

    # Generate and print report
    report = BenchmarkReport(0, results)
    io.print_report(test_case, report)
