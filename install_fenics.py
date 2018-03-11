import os
import argparse
import virtualenv

def clone_repos():
	"""
	cd "${SRC_DIR}"
	git clone https://bitbucket.org/fenics-project/fiat.git
	git clone https://bitbucket.org/fenics-project/ufl.git
	git clone https://bitbucket.org/fenics-project/dijitso.git
	git clone https://bitbucket.org/fenics-project/ffc.git

	git clone https://github.com/blechta/tsfc.git
	git clone https://github.com/blechta/COFFEE.git
	git clone https://github.com/blechta/FInAT.git

	git clone https://bitbucket.org/fenics-project/dolfin.git
	git clone https://github.com/pybind/pybind11.git
	"""
	pass

def pip_install():
	"""
	cd "${SRC_DIR}/fiat" && pip3 install -e .
	cd "${SRC_DIR}/ufl" && pip3 install -e .
	cd "${SRC_DIR}/dijitso" && pip3 install -e .
	cd "${SRC_DIR}/ffc" && pip3 install -e .

	cd "${SRC_DIR}/tsfc" && pip3 install -e .
	cd "${SRC_DIR}/COFFEE" && pip3 install -e .
	cd "${SRC_DIR}/FInAT" && pip3 install -e .

	pip3 install six singledispatch pulp
	"""

	"""
	# pip install a package using the venv as a prefix
	pip.main(["install", "--prefix", venv_dir, "xmltodict"])
	import xmltodict
	"""

	pass

def pybind_build():
	"""
	mkdir -p "${BUILD_DIR}/pybind11"
	cd "${BUILD_DIR}/pybind11"

	cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX="${PYBIND_DIR}" "${SRC_DIR}/pybind11"
	if ! make install
	then
		echo "Error when installing pybind11!"
		return 1
	fi

	return 0
	"""

	pass

def dolfin_build():
	"""
	mkdir "${BUILD_DIR}/dolfin"
	cd "${BUILD_DIR}/dolfin"

	cmake -DCMAKE_INSTALL_PREFIX="${DOLFIN_DIR}" "${SRC_DIR}/dolfin"
	if ! (make -j4 && make install)
	then
		echo "Error while building dolfin!"
		return 1
	fi

	cd "${SRC_DIR}/dolfin/python"
	export PYBIND11_DIR="${PYBIND_DIR}"
	export DOLFIN_DIR="${DOLFIN_DIR}"
	pip3 install -e .

	return 0
	"""

	pass

parser = argparse.ArgumentParser(description="install and set up an environment for FEniCS from git")
parser.add_argument("-r", "--repo-dir", type=str, help="Directory where source directories (repositories) will be stored.")
group = parser.add_mutually_exclusive_group()
group.add_argument("-l", "--local", action="store_true", help="Don't clone any repositories, use local files.")
group.add_argument("-co", "--clone-only", action="store_true", help="Only clone the required repositories, no install/build.")
#group.add_argument("-dr", "--dolfin-rebuild-only", action="store_true", help="only perform a dolfin rebuild/update")
parser.add_argument("install_prefix", type=str, help="FEniCS will be installed into this folder. If not specified, the current folder will be used.", nargs='?')
args = parser.parse_args()

FENICS_DIR = args.install_prefix
SRC_DIR = args.repo_dir
BUILD_DIR = None

if FENICS_DIR is None:
	FENICS_DIR = os.getcwd()

if SRC_DIR is None:
	SRC_DIR = os.path.join(FENICS_DIR, "src")

print(args)
print(FENICS_DIR)
print(SRC_DIR)

VENV_DIR = os.path.join(FENICS_DIR, "fenics_env")
BUILD_DIR = os.path.join(FENICS_DIR, "build")
PYBIND_DIR = os.path.join(FENICS_DIR, "include", "pybind11")
DOLFIN_DIR = os.path.join(FENICS_DIR, "dolfin")

if not os.path.exists(FENICS_DIR):
    os.makedirs(FENICS_DIR)

if not os.path.exists(SRC_DIR):
    os.makedirs(SRC_DIR)


# Create and activate the virtual environment
virtualenv.create_environment(VENV_DIR)
exec(open(os.path.join(VENV_DIR, "bin", "activate_this.py")).read())

clone_repos()
pip_install()

pybind_build()
dolfin_build()

#print(pip.main(["list"]))

#pip3 list

print("")
print(f"Activate FEniCS python virtualenv using 'source {FENICS_DIR}/fenics_env/bin/activate'")
print(f"Activate DOLFIN build environment using 'source ${FENICS_DIR}/dolfin/share/dolfin/dolfin.conf'")