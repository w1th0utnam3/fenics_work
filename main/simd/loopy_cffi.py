import numpy as np
import loopy as lp
import cffi
import importlib


def generate_kernel():
    knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]",
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        target=lp.CTarget())

    knl = lp.add_and_infer_dtypes(knl, {"a": np.dtype(np.double)})
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
    kernel_mod = importlib.import_module(kernel_name)
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    x = np.array([0, 1, 2, 3], dtype=np.double)
    y = np.array([0, 0, 0, 0], dtype=np.double)

    lib.loopy_kernel(ffi.cast("double*", x.ctypes.data),
                     4,
                     ffi.cast("double*", y.ctypes.data))

    print(y)

    result = np.allclose(y, 2*x)
    assert result
    return result


def run_test():
    kernel_name = "_kernel"

    kernel_c, kernel_h = generate_kernel()
    kernel_c, kernel_h = postprocess_loopy(kernel_c, kernel_h)

    print(kernel_c)
    print(kernel_h)

    compile_kernel(kernel_name, kernel_c, kernel_h)
    return test_kernel(kernel_name)
