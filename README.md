# FEniCS 


## Installing FEniCS from source
### Required packages
The following packages are required for FEniCS components
 - python3
 - python3-pip
 - libboost-all-dev
 - libeigen3-dev

As a single command:
```
sudo apt install python3 python3-pip libboost-all-dev libeigen3-dev
```

### Install script
The `install_fenics.py` script downloads all major FEniCS components (fiat, ufl, dijisto, ffc, dolfin and Firedrake's tsfc, coffee, finat) from their repositories (master branch) and installs them in a python virtual environment.

#### Options
Currently, the script has the following options:
```
usage: install_fenics.py [-h] [-r REPO_DIR] [-y] [-l | -co] [-j JOBS]
                         [install_prefix]

install and set up an environment for FEniCS from git

positional arguments:
  install_prefix        FEniCS will be installed into this folder. If not
                        specified, the current folder will be used.

optional arguments:
  -h, --help            show this help message and exit
  -r REPO_DIR, --repo-dir REPO_DIR
                        Directory where source directories (repositories) will
                        be stored/are already located.
  -y, --yes             Respond with 'yes' to all confirmation messages.
  -l, --local           Don't clone any repositories, use local files.
  -co, --clone-only     Only clone the required repositories, no
                        install/build.
  -j JOBS, --jobs JOBS  Number of make jobs to issue for building DOLFIN ('-j'
                        parameter for make). Default is to use
                        'os.cpu_count()'.
```
The `install_prefix` will contain the `fenics_env` virtual environment, the `dolfin` install directory, etc.

#### Virtual environment
In order to run the script, the `virtualenv` and `click` python package is required. Install it using:
```
pip3 install virtualenv click
```
Optionally, add the user python bin directory to your path in `~/.profile`:
```
export PATH=$PATH:~/.local/bin
```

### Optional packages

DOLFIN may make use of the following optional packages:
 * MPI, Message Passing Interface (MPI)
   Enables DOLFIN to run in parallel with MPI
 * PETSc (required version >= 3.7), Portable, Extensible Toolkit for Scientific Computation, <https://www.mcs.anl.gov/petsc/>
   Enables the PETSc linear algebra backend
 * SLEPc (required version >= 3.7), Scalable Library for Eigenvalue Problem Computations, <http://slepc.upv.es/>
 * SCOTCH, Programs and libraries for graph, mesh and hypergraph partitioning, <https://www.labri.fr/perso/pelegrin/scotch>
   Enables parallel graph partitioning
 * UMFPACK, Sparse LU factorization library, <http://faculty.cse.tamu.edu/davis/suitesparse.html>
 * BLAS, Basic Linear Algebra Subprograms, <http://netlib.org/blas/>
 * Threads
 * CHOLMOD, Sparse Cholesky factorization library for sparse matrices, <http://faculty.cse.tamu.edu/davis/suitesparse.html>
 * HDF5, Hierarchical Data Format 5 (HDF5), <https://www.hdfgroup.org/HDF5>
 * ZLIB, Compression library, <http://www.zlib.net>
They can be installed using:
```
sudo apt install zlib1g-dev libhdf5-dev petsc-dev slepc-dev
```

## Activating the environment
```
source fenics/fenics_env/bin/activate
source fenics/dolfin/share/dolfin/dolfin.conf
```