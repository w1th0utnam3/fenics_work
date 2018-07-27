import sys

import check_install

dir = "/local/fenics_work/submodules"
check_install.clean(dir)
if check_install.check(dir) != 0:
    print("Warning: Missing package(s) was/were installed.")
    sys.exit(1)
