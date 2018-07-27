import os
import sys
import inspect
import subprocess
import shutil


def check(source_dir: str):
    '''Checks whether all required packages are installed and installs them if necessary.'''
    print("Checking whether required packages are installed...")

    if not os.path.exists(source_dir):
        raise RuntimeError(f"Package src folder '{source_dir}' does not exist!")

    required_pkgs = [("fenics-ufl", "ufl"),
                     ("fenics-fiat", "fiat"),
                     ("fenics-dijitso", "dijitso"),
                     ("fenics-ffc", "ffcx"),
                     ("fenics-dolfin", "dolfinx/python")]

    packages = subprocess.check_output(["pip3", "list"]).decode('utf-8')
    #print(packages)

    missing = []
    for pkg_name, pkg_path in required_pkgs:
        if pkg_name not in packages:
            missing.append(pkg_path)

    if len(missing) == 0:
        print("All packages are installed.")
        return 0

    print("The following packages are missing:")
    print(missing)
    print("")

    for pkg_path in missing:
        print(f"Installing {pkg_path}...")
        output = subprocess.check_output(["pip3", "install", "-e", os.path.join(source_dir, pkg_path)],
                                         stderr=subprocess.STDOUT).decode('utf-8')
        print(output)

    return len(missing)


def clean(source_dir: str):
    '''Removes existing .egg and build directories.'''

    if not os.path.exists(source_dir):
        raise RuntimeError(f"Package src folder '{source_dir}' does not exist!")

    print("Removing existing .egg folders")

    folders = ["dolfinx/python/build",
               "dolfinx/python/fenics_dolfin.egg-info",
               "dijitso/fenics_dijitso.egg-info",
               "ffcx/fenics_ffc.egg-info",
               "fiat/fenics_fiat.egg-info",
               "ufl/fenics_ufl.egg-info"]

    for folder in folders:
        print(f"Removing '{folder}'...")
        shutil.rmtree(os.path.join(source_dir, folder), ignore_errors=True)

    print("Done.")


if __name__ == '__main__':
    # If this script is directly run, assume packages are located one dir higher
    script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))

    if len(sys.argv) > 1 and sys.argv[1] == "-c":
        clean(os.path.join(script_dir, "../"))

    check(os.path.join(script_dir, "../"))
