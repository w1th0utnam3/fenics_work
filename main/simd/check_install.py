import os
import subprocess

def check(source_dir: str):
    '''Checks whether all required packages are installed and install them if necessary.'''

    required_pkgs = [("fenics-ufl", "ufl"),
                     ("fenics-fiat", "fiat"),
                     ("fenics-dijitso", "dijitso"),
                     ("fenics-ffc", "ffcx"),
                     ("fenics-dolfin", "dolfinx/python"),
					 ("loo.py", "loopy")]

    packages = subprocess.check_output(["pip3", "list"]).decode('utf-8')

    missing = []
    for pkg_name, pkg_path in required_pkgs:
        if pkg_name not in packages:
            missing.append(pkg_path)

    for pkg_path in missing:
        print(f"Installing {pkg_path}...")
        output = subprocess.check_output(["pip3", "install", "-e", os.path.join(source_dir, pkg_path)]).decode('utf-8')
        print(output)

    return len(missing)
