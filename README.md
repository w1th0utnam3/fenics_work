# SIMD FEniCS project

Working repository for the [FEniCS Google Summer of Code SIMD project](https://flgsoc18.wordpress.com/2018/05/13/excited-for-fenics-and-gsoc/). Most code can be found in the [main/simd](main/simd) folder.

## Development environment

I perform most of the development on Windows using Docker. A longer introduction can be found on [my blog](https://flgsoc18.wordpress.com/2018/05/20/development-environment/) but for a quick start, the following sections should provide enough information.

### Docker image *(recommended)*

This repository provides a Dockerfile with all required components. In the root folder of the repository run
```
docker build -t simd-work .
```
or alternatively get it from the [quay.io repository](https://quay.io/repository/w1th0utnam3/simd-work) using
```
docker pull quay.io/w1th0utnam3/simd-work
```
then start container with
```
docker run -v [host path to repo]/fenics_work:/local/fenics_work -it simd-work /bin/bash
```
For development, the submodules in this repository (Dolfin-X, FFC-X, etc.) are mounted into the container and should be installed with `pip` in *editable* mode. To allow this, some `egg-info` folders have to be generated in the repositories. This has to be done once after first cloning this repository and is performed by the [`check_install.py`](main/simd/check_install.py) script:
```
docker run -v [host path to repo]/fenics_work:/local/fenics_work -it simd-work python3 /local/fenics_work/main/simd/check_install.py
```

### Debugging

On Ubuntu the default core file size limit is 0. Use
```
ulimit -c unlimited
```
to remove this restrictions. However this has to be rerun in every shell session.
The core dumps will be located in the working directory. They can be inspected by launching
```
gdb <executable path> <core dump path>
```
followed by the `bt` (backtrace) command inside `gdb`. The command `quit` can be used to exit `gdb`.
