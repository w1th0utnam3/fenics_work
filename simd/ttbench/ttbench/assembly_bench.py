import time
import numpy as np

import ufl
import dolfin
import dolfin.fem.assembling
from dolfin import MPI

import ttbench.forms_dolfin as raw_forms

MESH_SIZE = 30
BENCH_REPEATS = 4

# FIXME: Benchmarking of just 1-forms?


def forms():
    mesh = dolfin.UnitCubeMesh(MPI.comm_world, MESH_SIZE, MESH_SIZE, MESH_SIZE)
    cell = mesh.ufl_cell()

    el_p1 = ufl.FiniteElement("Lagrange", cell, 1)
    el_p2 = ufl.FiniteElement("Lagrange", cell, 2)
    vec_el_p1 = ufl.VectorElement("Lagrange", cell, 1)

    laplace = ("Laplace P1", mesh, raw_forms.laplace_forms(mesh, el_p1))
    laplace_p2p1 = ("Laplace P2,P1 coeff", mesh, raw_forms.laplace_coeff_forms(mesh, el_p2, el_p1))
    hyperelasticity = ("Hyperelasticity", mesh, raw_forms.hyperelasticity_forms(mesh, vec_el_p1))

    return [
        laplace,
        laplace_p2p1,
        hyperelasticity
    ]


def run_benchmark(a, L, bc, form_compiler_parameters):
    bcs = [bc] if bc else []
    assembler = dolfin.fem.assembling.Assembler([[a]], [L], bcs, form_compiler_parameters)

    t = -time.time()
    A, b = assembler.assemble(mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    t += time.time()

    return A, b, t


def get_ffc_params():
    ffc_param_default = {
        "cpp_optimize_flags": "-O2"
    }

    ffc_param_default_vec = {
        "cpp_optimize_flags": "-O2 -ftree-vectorize -march=native -mtune=native"
    }

    ffc_param_batch_gcc_ext = {
        "cell_batch_size": 4,
        "enable_cross_cell_gcc_ext": True,
        "cpp_optimize_flags": "-O2 -funroll-loops -ftree-vectorize -march=native -mtune=native"
    }

    return [
        ("FFC default (reference)", ffc_param_default),
        ("FFC default with auto vec", ffc_param_default_vec),
        ("FFC cross-cell with auto vec", ffc_param_batch_gcc_ext)
    ]


def run_assembly_bench():
    print("Assembly benchmark")
    print("-" * 20)
    print("")

    indent_level = 0

    def print_indent(msg):
        nonlocal indent_level
        print("".join(["\t"]*indent_level + [msg]))

    # Loop over all forms/problems to benchmark
    n_repeats = BENCH_REPEATS
    for problem_name, mesh, (a, L, bc) in forms():
        print_indent("Benchmarking {} forms...".format(problem_name))
        print_indent("")

        A_ref, b_ref = None, None
        A_ref_norm, b_ref_norm = 0, 0
        t_ref_avg = 0

        indent_level += 1

        t_form = -time.time()
        # Loop over all FFC+dijitso parameter sets
        for i, (param_name, param_set) in enumerate(get_ffc_params()):
            print_indent(param_name)

            print_indent("1. Checking correctness...")
            indent_level += 1

            # Make first assembly run for correctness checking
            A, b, t = run_benchmark(a, L, bc, param_set)
            print_indent("Assembly time: {:.2f}ms".format(t * 1000))

            if i == 0:
                # Store reference values
                A_ref = A
                b_ref = b

                A_ref_norm = A_ref.norm(dolfin.cpp.la.Norm.frobenius)
                b_ref_norm = b_ref.norm(dolfin.cpp.la.Norm.l2)

                print_indent("norm(A)={:.14e}".format(A_ref_norm))
                print_indent("norm(b)={:.14e}".format(b_ref_norm))
            else:
                # Compare results with reference values
                A_norm = A.norm(dolfin.cpp.la.Norm.frobenius)
                b_norm = b.norm(dolfin.cpp.la.Norm.l2)

                print_indent("norm(A)={:.14e}, ok: {}".format(A_norm, np.isclose(A_norm, A_ref_norm)))
                print_indent("norm(b)={:.14e}, ok: {}".format(b_norm, np.isclose(b_norm, b_ref_norm)))

            print_indent("")
            indent_level -= 1

            print_indent("2. Timing...")
            indent_level += 1

            # Make multiple timed assembly runs
            times = []
            for j in range(n_repeats):
                A, b, t = run_benchmark(a, L, bc, param_set)
                times.append(t)

            # Calculate mean assembly time and speedup
            t_avg = np.mean(times)
            if i == 0:
                t_ref_avg = t_avg
            speedup = t_ref_avg / t_avg

            # All values that will be printed
            values = [
                n_repeats,
                np.min(times) * 1000,
                np.max(times) * 1000,
                t_avg * 1000,
                speedup
            ]

            # Print timing results
            print_indent("n={}, min={:.2f}ms, max={:.2f}ms, avg={:.2f}ms, speedup: {:.2f}x".format(*values))

            indent_level -= 1
            print_indent("-" * 20)
            print_indent("")
        t_form += time.time()
        print_indent("Total time for benchmarking this problem: {:.2f}ms".format(t_form * 1000))
        print_indent("")

        indent_level -= 1
