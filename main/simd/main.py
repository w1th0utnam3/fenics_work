import sys
import check_install

if check_install.check("/local/fenics_work/main") != 0:
    print("Warning: Missing package was installed. Please rerun script.")
    sys.exit(1)

import simple_examples
import loopy_cffi
import custom_kernel


def main():
    #simple_examples.loopy_example()
    #simple_examples.poisson_example()

    loopy_cffi.run_example()
    custom_kernel.run_example()

    return

if __name__ == '__main__':
    main()