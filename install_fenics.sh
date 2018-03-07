#!/bin/sh

mkdir ~/fenics
cd ~/fenics

virtualenv fenics_env
. ./fenics_env/bin/activate

mkdir src
cd src

git clone https://bitbucket.org/fenics-project/fiat.git
git clone https://bitbucket.org/fenics-project/ufl.git
git clone https://bitbucket.org/fenics-project/dijitso.git
git clone https://bitbucket.org/fenics-project/ffc.git

git clone https://github.com/blechta/tsfc.git
git clone https://github.com/blechta/COFFEE.git
git clone https://github.com/blechta/FInAT.git

cd fiat && pip3 install -e . && cd ..
cd ufl && pip3 install -e . && cd ..
cd dijitso && pip3 install -e . && cd ..
cd ffc && pip3 install -e . && cd ..

cd tsfc && pip3 install -e . && cd ..
cd COFFEE && pip3 install -e . && cd ..
cd FInAT && pip3 install -e . && cd ..

cd ..

echo
pip3 list

echo
echo "Activate the virtualenv using 'source fenics_env/bin/activate'"
