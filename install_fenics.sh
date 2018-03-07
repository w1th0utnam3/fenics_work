#!/bin/sh

mkdir fenics
cd fenics

virtualenv fenics_env
. ./fenics_env/bin/activate

mkdir src
cd src

git clone git@bitbucket.org:fenics-project/fiat
git clone git@bitbucket.org:fenics-project/ufl
git clone git@bitbucket.org:fenics-project/dijitso
git clone git@bitbucket.org:fenics-project/ffc

cd fiat && pip3 install -e . && cd ..
cd ufl && pip3 install -e . && cd ..
cd dijitso && pip3 install -e . && cd ..
cd ffc && pip3 install -e . && cd ..
cd ..

echo
pip3 list

echo
echo "Activate the virtualenv using 'source fenics_env/bin/activate'"
