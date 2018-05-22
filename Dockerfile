# Dockerfile to build the FEniCS-X development libraries
#
# Authors:
# Jack S. Hale <jack.hale@uni.lu>
# Lizao Li <lzlarryli@gmail.com>
# Garth N. Wells <gnw20@cam.ac.uk>
# Jan Blechta <blechta@karlin.mff.cuni.cz>

FROM fenicsproject/dolfinx:latest
LABEL maintainer="fenics-project <fenics-support@googlegroups.org>"

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
ARG DOLFINX_COMMIT_HASH=16ac4c36913d197d7f364b347b40a528188e401b
RUN git clone --recurse-submodules https://github.com/FEniCS/dolfinx.git && \
	cd dolfinx && \
	git checkout ${DOLFINX_COMMIT_HASH} && \
	mkdir build && \
	cd build && \
	cmake ../cpp && \
	make -j8 && \
	make install

# Install dolfinx python package
RUN pip3 install -e /local/fenics/dolfinx/python

# Install packages for TSFC code generatoin
RUN pip3 install six singledispatch pulp networkx
RUN pip3 install --no-cache-dir git+https://github.com/blechta/tsfc.git
RUN pip3 install --no-cache-dir git+https://github.com/blechta/coffee.git
RUN pip3 install --no-cache-dir git+https://github.com/blechta/finat.git
