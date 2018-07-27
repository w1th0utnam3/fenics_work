import os
import cffi
import importlib

import numpy as np

import simd.utils as utils

def compile(module_name: str, verbose: bool = False):

    # Additional compiler arguments
    compile_args = {
        "-O2": True,
        "-funroll-loops": True,
        "-ftree-vectorize": True,
        "-march=native": True,
        "-mtune=native": True,
        "-fopenmp": True
    }
    # Remove disabled args
    active_compile_args = [arg for arg, use in compile_args.items() if use]

    cur_path = os.path.dirname(os.path.abspath(__file__))
    def get_file_as_str(filename: str):
        with open(os.path.join(cur_path, filename), "r") as file:
            strg = file.read()
        return strg

    code_c, code_h = "", ""

    code_c += get_file_as_str("full_block_test_elem.c")
    code_c += get_file_as_str("full_block_test_avx.c")
    code_c += get_file_as_str("full_block_test_ffc.c")
    code_c += get_file_as_str("full_block_test_ffc_padded.c")
    code_c += get_file_as_str("full_block_test.c")

    code_h += get_file_as_str("full_block_test.h")

    # Build the kernel
    print(f"Compiling with args: '{', '.join(active_compile_args)}'")

    ffi = cffi.FFI()
    ffi.set_source(module_name, code_c, extra_compile_args=active_compile_args)
    ffi.cdef(code_h)
    ffi.compile(verbose=verbose)

    print("Finished compiling kernel.")

    # Import the compiled kernel
    kernel_mod = importlib.import_module(f"simd.tmp.{module_name}")
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    print("Imported kernel.")

    return ffi, lib


def run_example():
    # Mesh size, (n+1)^3 vertices
    n = int(np.floor(120**3/4)*4)
    n_runs = 5

    ffi, lib = compile("_full_block_timing", verbose=False)

    # Check that the functions calculate the same values
    assert (np.isclose(lib.call_tabulate_ffc(1), lib.call_tabulate_avx(1)))
    assert (np.isclose(lib.call_tabulate_ffc(4), lib.call_tabulate_elem(4)))
    assert (np.isclose(lib.call_tabulate_ffc(2), lib.call_tabulate_ffc_padded(2)))

    # Define the kernel generators
    kernels = {
        "ffc"        : lambda: lib.call_tabulate_ffc(n),
        "ffc_padded" : lambda: lib.call_tabulate_ffc_padded(n),
        "manual_avx" : lambda: lib.call_tabulate_avx(n),
        "elem"       : lambda: lib.call_tabulate_elem(n)
    }

    name_length = max([len(kernel) for kernel in kernels.keys()])
    results = {kernel : utils.timing(n_runs, test_callable, verbose=True) for kernel, test_callable in kernels.items()}

    print("")
    reference_time = 3168.84
    print(f"Runtime of tablute_tensor calls, mesh with {n} elements, average over {n_runs} runs\navg/min/max in ms, speedup:")
    for kernel, result in results.items():
        time_avg, time_min, time_max = result
        print(
            f"{kernel:<{name_length+1}}\t"
            f"{time_avg*1000:>8.2f}\t"
            f"{time_min*1000:>8.2f}\t"
            f"{time_max*1000:>8.2f}\t"
            f"{round(reference_time/(time_avg*1000),2)}x")
