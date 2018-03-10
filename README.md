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
The `install_fenics.sh` script downloads all major FEniCS components (fiat, ufl, dijisto, ffc, dolfin and Firedrake's tsfc, coffee, finat) from their repositories (master branch) and installs them in a python virtual environment.

#### Options
The supports a single optional argument which specifies the install prefix for the script. In this prefix another folder `fenics` is created, i.e.
```
install_fenics.sh ~
```
creates the folder `fenics` in the user's home directory and installs the components inside of it

#### Virtual environment
In order to run the script, the `virtualenv` python package is required. Install it using:
```
pip3 install virtualenv
```
Optionally add the user python bin directory to your path in `~/.profile`:
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
