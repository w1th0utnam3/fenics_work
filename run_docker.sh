#! /bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
docker run --cap-add=SYS_PTRACE --rm -it -v${DIR}:/local/fenics_work quay.io/w1th0utnam3/simd-work:latest /bin/bash
