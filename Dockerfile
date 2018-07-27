# Dockerfile to build the ffcx simd dev env
#

FROM quay.io/w1th0utnam3/docker-dolfinx-base:latest

# Install other dependencies
RUN pip3 install cffi pytest

# Clone this repository
WORKDIR /local
RUN git clone --recurse-submodules https://github.com/w1th0utnam3/fenics_work.git

# Install FIAT, UFL, dijitso and ffcx
RUN pip3 install -e /local/fenics_work/main/fiat && \
	pip3 install -e /local/fenics_work/main/ufl && \
	pip3 install -e /local/fenics_work/main/dijitso && \
	pip3 install -e /local/fenics_work/main/ffcx

# Build and install dolfinx
RUN cd /local/fenics_work/main/dolfinx && \
	mkdir build && \
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=Developer ../cpp && \
	make -j8 && \
	make install

# Install dolfinx python package
RUN pip3 install -e /local/fenics_work/main/dolfinx/python

WORKDIR /tmp

# Install dependencies for TSFC code generation
RUN pip3 install six singledispatch pulp networkx

# Install packages for TSFC code generation
RUN pip3 install --no-cache-dir git+https://github.com/firedrakeproject/tsfc.git
RUN pip3 install --no-cache-dir git+https://github.com/coneoproject/COFFEE.git
RUN pip3 install --no-cache-dir git+https://github.com/FInAT/FInAT.git

# Additional packages for development
RUN pip3 install dataclasses

WORKDIR /local/fenics_work/main
