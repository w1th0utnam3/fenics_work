#!/bin/sh

if [ $# -gt 0 ]; then
	INSTALL_PREFIX=$1
else
	INSTALL_PREFIX="$(pwd)"
fi

echo "Using ${INSTALL_PREFIX} as install prefix."

FENICS_DIR="${INSTALL_PREFIX}/fenics"
SRC_DIR="${FENICS_DIR}/src"
PYBIND_DIR="${FENICS_DIR}/pybind11"
DOLFIN_DIR="${FENICS_DIR}/dolfin"

mkdir "${FENICS_DIR}"
mkdir "${SRC_DIR}"

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

cd "${FENICS_DIR}"

virtualenv fenics_env
. ./fenics_env/bin/activate

cd "${SRC_DIR}"

cd fiat && pip3 install -e . && cd ..
cd ufl && pip3 install -e . && cd ..
cd dijitso && pip3 install -e . && cd ..
cd ffc && pip3 install -e . && cd ..

cd tsfc && pip3 install -e . && cd ..
cd COFFEE && pip3 install -e . && cd ..
cd FInAT && pip3 install -e . && cd ..

pip3 install six singledispatch pulp

cd "${SRC_DIR}/pybind"
mkdir build
cd build
cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX="${PYBIND_DIR}" ../
make install

cd "${SRC_DIR}/dolfin"
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="${DOLFIN_DIR}" ../
make -j4
make install

# TODO: The following is not working, because of paths!
cd "${SRC_DIR}/dolfin/python"
export PYBIND11_DIR="${PYBIND_DIR}"
export DOLFIN_DIR="${DOLFIN_DIR}"
pip3 install -e .

echo
pip3 list

echo
echo "Activate the virtualenv using 'source fenics_env/bin/activate'"
