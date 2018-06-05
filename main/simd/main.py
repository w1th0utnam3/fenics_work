import sys
import check_install

if check_install.check("/local/fenics_work/main") != 0:
    print("Warning: Missing package(s) was/were installed. Please rerun script.")
    sys.exit(1)

import simple_examples
import loopy_cffi
import poisson_2d.custom_kernel
import poisson_2d.custom_kernel_loopy
import poisson_3d.custom_kernel_loopy_3d_simple
import poisson_3d.poisson_blog_post_numba
import poisson_3d.poisson_blog_post_c
import poisson_3d.poisson_tensor_mode


def main():
    #simple_examples.loopy_example()
    #simple_examples.poisson_example()

    poisson_3d.poisson_tensor_mode.run_example()

    return

if __name__ == '__main__':
    main()