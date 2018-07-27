import simd.simple_examples
import simd.loopy_cffi
import simd.poisson_3d.poisson_tensor_mode
import simd.full_block_test.timing_form


def run_examples():
    #simd.simple_examples.loopy_example()
    #simd.simple_examples.poisson_example()
    #elasticity()

    #simd.poisson_3d.poisson_tensor_mode.run_example()
    simd.full_block_test.timing_form.run_example()

    #poisson_3d.poisson_tsfc_coffee.run_example()

    return