import numpy

import dolfin
import dolfin.fem.assembling
from dolfin import MPI

import ufl
from ufl import dot, grad, dx, ds, tr, det, ln

import time


def assemble_test(cell_batch_size: int):
    mesh = dolfin.UnitCubeMesh(MPI.comm_world, 60, 60, 60)
    cell = mesh.ufl_cell()

    vec_element = dolfin.VectorElement("Lagrange", cell, 1)
    # scl_element = dolfin.FiniteElement("Lagrange", cell, 1)

    Q = dolfin.FunctionSpace(mesh, vec_element)
    # Qs = dolfin.FunctionSpace(mesh, scl_element)

    # Coefficients
    v = dolfin.function.argument.TestFunction(Q)  # Test function
    du = dolfin.function.argument.TrialFunction(Q)  # Incremental displacement
    u = dolfin.Function(Q)  # Displacement from previous iteration

    B = dolfin.Constant((0.0, -0.5, 0.0), cell)  # Body force per unit volume
    T = dolfin.Constant((0.1, 0.0, 0.0), cell)  # Traction force on the boundary

    # B, T = dolfin.Function(Q), dolfin.Function(Q)

    # Kinematics
    d = u.geometric_dimension()
    F = ufl.Identity(d) + grad(u)  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J = det(F)

    # Elasticity parameters
    E, nu = 10.0, 0.3
    mu = dolfin.Constant(E / (2 * (1 + nu)), cell)
    lmbda = dolfin.Constant(E * nu / ((1 + nu) * (1 - 2 * nu)), cell)

    # mu, lmbda = dolfin.Function(Qs), dolfin.Function(Qs)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu / 2) * (Ic - 3) - mu * ln(J) + (lmbda / 2) * (ln(J)) ** 2

    # Total potential energy
    Pi = psi * dx - dot(B, u) * dx - dot(T, u) * ds

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = ufl.derivative(Pi, u, v)

    # Compute Jacobian of F
    J = ufl.derivative(F, u, du)

    a, L = J, F

    if cell_batch_size > 1:
        cxx_flags = "-O2 -ftree-vectorize -funroll-loops -march=native -mtune=native"
    else:
        cxx_flags = "-O2"

    assembler = dolfin.fem.assembling.Assembler([[a]], [L], [],
                                                form_compiler_parameters={"cell_batch_size": cell_batch_size,
                                                                          "enable_cross_element_gcc_ext": True,
                                                                          "cpp_optimize_flags": cxx_flags})

    t = -time.time()
    A, b = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    t += time.time()

    return A, b, t


A1, b1, t1 = assemble_test(cell_batch_size=1)
print("{:.4f}s".format(t1))

A4, b4, t4 = assemble_test(cell_batch_size=4)
print("{:.4f}s".format(t4))
print("")

A1norm = A1.norm(dolfin.cpp.la.Norm.frobenius)
b1norm = b1.norm(dolfin.cpp.la.Norm.l2)

A4norm = A4.norm(dolfin.cpp.la.Norm.frobenius)
b4norm = b4.norm(dolfin.cpp.la.Norm.l2)

print(A1norm)
print(A4norm)
print(b1norm)
print(b4norm)

assert(numpy.isclose(A1norm, A4norm))
assert(numpy.isclose(b1norm, b4norm))
