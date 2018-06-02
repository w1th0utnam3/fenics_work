import sys
import check_install

if check_install.check("/local/fenics_work/main") != 0:
    print("Warning: Missing package(s) was/were installed. Please rerun script.")
    sys.exit(1)

import simple_examples
import loopy_cffi
import custom_kernel
import custom_kernel_loopy


def main():
    #simple_examples.loopy_example()
    #simple_examples.poisson_example()

    #loopy_cffi.run_example()
    #custom_kernel.run_example()
    custom_kernel_loopy.run_example()

    return

if __name__ == '__main__':
    main()