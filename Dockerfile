# Dockerfile to build the ffcx simd dev env
#

FROM quay.io/w1th0utnam3/dolfinx:latest

WORKDIR /tmp

# Install OpenCL loader
RUN apt-get -qq update && \
	apt-get -y --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
	apt-get -y install \
	ocl-icd-opencl-dev && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Install PyOpenCL
RUN pip3 install --no-cache-dir pyopencl

WORKDIR /local/fenics

RUN git clone --recurse-submodules https://github.com/inducer/loopy.git && \
	pip3 install -e /local/fenics/loopy

# Install FIAT, UFL, dijitso and ffcx (development versions, master branch)
RUN git clone --recurse-submodules https://bitbucket.org/fenics-project/fiat.git && \
	git clone --recurse-submodules https://bitbucket.org/fenics-project/ufl.git && \
	git clone --recurse-submodules https://bitbucket.org/fenics-project/dijitso.git && \
	git clone --recurse-submodules https://github.com/w1th0utnam3/ffcx.git && \
	pip3 install -e /local/fenics/fiat && \
	pip3 install -e /local/fenics/ufl && \
	pip3 install -e /local/fenics/dijitso && \
	pip3 install -e /local/fenics/ffcx

# Build and install dolfinx, use the commit that was used to build the base image
ARG DOLFINX_COMMIT_HASH=ece4f4905758931ff4b34d3d3fce745c6acef64e
RUN git clone --recurse-submodules https://github.com/w1th0utnam3/dolfinx.git && \
	cd dolfinx && \
	git checkout ${DOLFINX_COMMIT_HASH} && \
	mkdir build && \
	cd build && \
	cmake ../cpp && \
	make -j8 && \
	make install

# Install dolfinx python package
RUN pip3 install -e /local/fenics/dolfinx/python

WORKDIR /tmp

# Install dependencies for TSFC code generation
RUN pip3 install six singledispatch pulp networkx

# Install packages for TSFC code generation
RUN git clone https://github.com/firedrakeproject/tsfc.git && \
	cd tsfc && \
	git fetch && \
	git checkout tsfc2loopy && \
	cd / && \
	rm -rf /tmp/* /var/tmp/*

RUN pip3 install --no-cache-dir git+https://github.com/coneoproject/COFFEE.git
RUN pip3 install --no-cache-dir git+https://github.com/FInAT/FInAT.git

WORKDIR /local/fenics
