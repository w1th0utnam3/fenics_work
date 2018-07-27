#! /bin/bash

rm -rf /local/fenics_work/submodules/dolfinx/build
rm -rf /local/fenics_work/submodules/dolfinx/python/build
rm -rf /local/fenics_work/submodules/dolfinx/python/fenics_dolfin.egg-info

cd /local/fenics_work/submodules/dolfinx && \
	mkdir build && \
	cd build && \
	cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer ../cpp && \
	ninja -j8 install

pip3 install -e /local/fenics_work/submodules/dolfinx/python
