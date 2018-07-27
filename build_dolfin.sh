#! /bin/bash

cd /local/fenics_work/main/dolfinx/build && \
	make -j8 && \
	make install
