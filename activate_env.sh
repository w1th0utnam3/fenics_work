#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

export PYTHONPATH=${PYTHONPATH}:${DIR}/main
export PYTHONPATH=${PYTHONPATH}:${DIR}/main/dijitso
export PYTHONPATH=${PYTHONPATH}:${DIR}/main/dolfinx/python
export PYTHONPATH=${PYTHONPATH}:${DIR}/main/ffcx
export PYTHONPATH=${PYTHONPATH}:${DIR}/main/fiat
export PYTHONPATH=${PYTHONPATH}:${DIR}/main/minidolfin
export PYTHONPATH=${PYTHONPATH}:${DIR}/main/ufl
export PYTHONPATH=${PYTHONPATH}:${DIR}/main/simd/ttbench
