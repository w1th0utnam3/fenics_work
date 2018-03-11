#!/bin/bash

# Clones all required repositories
function clone_repos {
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
}

# Installs all python packages
function pip_install {
	# Install python packages
	cd "${SRC_DIR}/fiat" && pip3 install -e .
	cd "${SRC_DIR}/ufl" && pip3 install -e .
	cd "${SRC_DIR}/dijitso" && pip3 install -e .
	cd "${SRC_DIR}/ffc" && pip3 install -e .

	cd "${SRC_DIR}/tsfc" && pip3 install -e .
	cd "${SRC_DIR}/COFFEE" && pip3 install -e .
	cd "${SRC_DIR}/FInAT" && pip3 install -e .

	pip3 install six singledispatch pulp
}

function pybind_build {
	mkdir -p "${BUILD_DIR}/pybind11"
	cd "${BUILD_DIR}/pybind11"

	cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX="${PYBIND_DIR}" "${SRC_DIR}/pybind11"
	if ! make install
	then
		echo "Error when installing pybind11!"
		return 1
	fi

	return 0
}

function dolfin_build {
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
}

# Use first argument as install prefix
if (("$#" > 0))
then
	INSTALL_PREFIX=$1
else
	INSTALL_PREFIX="$(pwd)"
fi

echo "Using ${INSTALL_PREFIX} as install prefix."

FENICS_DIR="${INSTALL_PREFIX}/fenics"
SRC_DIR="${FENICS_DIR}/src"
BUILD_DIR="${FENICS_DIR}/build"
PYBIND_DIR="${FENICS_DIR}/include/pybind11"
DOLFIN_DIR="${FENICS_DIR}/dolfin"

# Try to create directory for fenics
mkdir -p "${SRC_DIR}"
if ! cd "${SRC_DIR}"
then
	echo "Unable to create/access directory ${SRC_DIR}!"
	exit 1
fi

# Try to create virtual environment and check if successful
cd "${FENICS_DIR}"
if ! virtualenv fenics_env
then
	echo "Could not create virtual environment!"
	echo "Is virtualenv installed and included in PATH?"
	exit 1
else
	. ./fenics_env/bin/activate
fi

clone_repos
pip_install

pybind_build
dolfin_build

echo
pip3 list

echo
echo "Activate FEniCS python virtualenv using 'source ${FENICS_DIR}/fenics_env/bin/activate'"
echo "Activate DOLFIN build environment using 'source ${FENICS_DIR}/dolfin/share/dolfin/dolfin.conf'"
