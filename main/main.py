import sys

from simd import check_install
from simd.run_examples import run_examples

dir = "/local/fenics_work/main"
#check_install.clean(dir)
if check_install.check(dir) != 0:
    print("Warning: Missing package(s) was/were installed. Please rerun script.")
    sys.exit(1)

def main():
    run_examples()

if __name__ == '__main__':
    main()