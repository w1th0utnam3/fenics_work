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

# Install OpenCL loader and packages required to install Intel OpenCL driver
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    ocl-icd-opencl-dev \
    rpm2cpio \
    cpio && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Intel OpenCL driver
ARG INTELCL_VERSION_SHORT=16.1.2
ARG INTELCL_VERSION_FULL=16.1.2_x64_rh_6.4.0.37
RUN cd /tmp && \
    wget http://registrationcenter-download.intel.com/akdlm/irc_nas/12556/opencl_runtime_${INTELCL_VERSION_FULL}.tgz && \
    tar -xvf opencl_runtime_${INTELCL_VERSION_FULL}.tgz && \
    cd ./opencl_runtime_* && \
    TGT_DIR="/opt/intel-opencl-icd-${INTELCL_VERSION_SHORT}/lib" && \
    mkdir -p "$TGT_DIR" && \
    rpm2cpio rpm/opencl-*-intel-cpu-*.x86_64.rpm | cpio -idmv && \
    cp ./opt/intel/opencl-*/lib64/* "$TGT_DIR" && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "$TGT_DIR/libintelocl.so" > /etc/OpenCL/vendors/intel.icd && \
    cd /tmp && \
    rm -rf /tmp/*

# Select the "Intel CPU device" added by the Intel driver as default PyOpenCL device
ENV PYOPENCL_CTX='0'
# Install PyOpenCL
RUN pip3 install --no-cache-dir pyopencl

WORKDIR /local/fenics

RUN git clone https://github.com/inducer/loopy.git && \
	pip3 install -e /local/fenics/loopy

# Install FIAT, UFL, dijitso and ffcx (development versions, master branch)
RUN git clone https://bitbucket.org/fenics-project/fiat.git && \
	git clone https://bitbucket.org/fenics-project/ufl.git && \
	git clone https://bitbucket.org/fenics-project/dijitso.git && \
	git clone https://github.com/w1th0utnam3/ffcx.git && \
	pip3 install -e /local/fenics/fiat && \
	pip3 install -e /local/fenics/ufl && \
	pip3 install -e /local/fenics/dijitso && \
	pip3 install -e /local/fenics/ffcx

# Build and install dolfinx
RUN git clone https://github.com/FEniCS/dolfinx.git && \
	cd dolfinx && \
	mkdir build && \
	cd build && \
	cmake ../cpp && \
	make && \
	make install

# Install dolfinx python package
RUN pip3 install -e /local/fenics/dolfinx/python
