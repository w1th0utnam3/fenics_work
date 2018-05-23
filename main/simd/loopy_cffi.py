import numpy as np
import loopy as lp
import cffi
import importlib

import time


def generate_kernel():
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]",
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        target=lp.CTarget())

    knl = lp.add_and_infer_dtypes(knl, {"a": np.dtype(np.double)})
    #knl = lp.split_iname(knl, "i", 4)
    #knl = lp.tag_inames(knl, dict(i_inner="unr"))

    return lp.generate_code_v2(knl).all_code(), str(lp.generate_header(knl)[0])


def postprocess_loopy(code_c: str, code_h: str):
    # CFFI does not support "__restrict__", therefore replace it with C99 "restrict"
    code_c = code_c.replace("__restrict__", "restrict")
    code_h = code_h.replace("__restrict__", "restrict")

    return code_c, code_h


def compile_kernel(kernel_name: str, kernel_c: str, kernel_h: str):
    ffi = cffi.FFI()

    ffi.set_source(kernel_name, kernel_c)
    ffi.cdef(kernel_h)
    lib = ffi.compile(verbose=True)

    return lib


def test_kernel(kernel_name: str):
    # Import the compiled kernel
    kernel_mod = importlib.import_module(kernel_name)
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    # Generate test data
    n = int(1e7)
    x = np.random.rand(n)
    # Reserve result storage
    y_lp = np.zeros_like(x)
    y_np = np.zeros_like(x)

    start_lp, end_lp, start_np, end_np = 0,0,0,0

    for i in range(2):
        # Run the loopy kernel
        start_lp = time.time()
        lib.loopy_kernel(ffi.cast("double*", x.ctypes.data),
                         n,
                         ffi.cast("double*", y_lp.ctypes.data))
        end_lp = time.time()

        # Run the corresponding numpy operation
        start_np = time.time()
        np.multiply(2, x, y_np)
        end_np = time.time()

    # Check results
    success = np.allclose(y_lp, y_np)
    print(f"Multiplying n={n} random doubles by 2.")
    print(f"L2 error between loopy and numpy: {np.linalg.norm(y_lp - y_np)}")
    print(f"Timing loopy: {(end_lp - start_lp)*1000}ms, numpy: {(end_np - start_np)*1000}ms")

    return success


def run_example():
    """Runs a Loopy + CFFI example"""

    kernel_name = "_kernel"

    kernel_c, kernel_h = generate_kernel()
    kernel_c, kernel_h = postprocess_loopy(kernel_c, kernel_h)

    print("Generated kernel code:")
    print(kernel_c)
    print("Generated kernel header:")
    print(kernel_h)
    print("\n")

    compile_kernel(kernel_name, kernel_c, kernel_h)
    print("\n")

    return test_kernel(kernel_name)
