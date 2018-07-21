import dolfin
import dolfin.cpp
import dolfin.cpp.fem
from dolfin import *
from dolfin.la import PETScMatrix, PETScVector, PETScKrylovSolver
from dolfin.jit.jit import ffc_jit

import numba as nb
import numpy as np

import sys
import time

# Generate a unit cube with (n+1)^3 vertices
n = 22
mesh = UnitCubeMesh(MPI.comm_world, n, n, n)
Q = FunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(Q)
v = TestFunction(Q)

# Define the boundary: vertices where any component is in machine precision accuracy 0 or 1
def boundary(x):
    return np.sum(np.logical_or(x < DOLFIN_EPS, x > 1.0 - DOLFIN_EPS), axis=1) > 0

u0 = Constant(0.0)
bc = DirichletBC(Q, u0, boundary)

# Initialize bilinear form and rhs
a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
L = dolfin.cpp.fem.Form([Q._cpp_object])

# Bilinear form
jit_result = ffc_jit(dot(grad(u), grad(v)) * dx,
                     form_compiler_parameters={"cell_batch_size": 4, "enable_cross_element_gcc_ext": True})
ufc_form = dolfin.cpp.fem.make_ufc_form(jit_result[0])
a = dolfin.cpp.fem.Form(ufc_form, [Q._cpp_object, Q._cpp_object])

# Rhs
f = Expression("2.0", element=Q.ufl_element())
jit_result = ffc_jit(f*v * dx,
                     form_compiler_parameters={"cell_batch_size": 4, "enable_cross_element_gcc_ext": True})
ufc_form = dolfin.cpp.fem.make_ufc_form(jit_result[0])
L = dolfin.cpp.fem.Form(ufc_form, [Q._cpp_object])
# Attach rhs expression as coefficient
L.set_coefficient(0, f._cpp_object)

assembler = dolfin.cpp.fem.Assembler([[a]], [L], [])

A = PETScMatrix()
b = PETScVector()

assembler.assemble(A, dolfin.cpp.fem.Assembler.BlockType.monolithic)
assembler.assemble(b, dolfin.cpp.fem.Assembler.BlockType.monolithic)

Anorm = A.norm(dolfin.cpp.la.Norm.frobenius)
bnorm = b.norm(dolfin.cpp.la.Norm.l2)
print(Anorm, bnorm)

# Norms obtained with FFC and n=22 and bcs
#assert (np.isclose(Anorm, 60.86192203436385))
#assert (np.isclose(bnorm, 0.018075523965828778))

# Norms obtained with FFC and n=22 and no bcs
assert (np.isclose(Anorm, 29.416127208482518))
assert (np.isclose(bnorm, 0.018726593629987284))

sys.exit(0)

comm = L.mesh().mpi_comm()
solver = PETScKrylovSolver(comm)

u = Function(Q)
solver.set_operator(A)
solver.solve(u.vector(), b)
