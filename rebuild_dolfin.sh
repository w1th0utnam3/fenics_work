#! /bin/bash

rm -rf /local/fenics_work/main/dolfinx/build
rm -rf /local/fenics_work/main/dolfinx/python/build
rm -rf /local/fenics_work/main/dolfinx/python/fenics_dolfin.egg-info

cd /local/fenics_work/main/dolfinx && \
	mkdir build && \
	cd build && \
	cmake ../cpp && \
	make -j8 && \
	make install

pip3 install -e /local/fenics_work/main/dolfinx/python
