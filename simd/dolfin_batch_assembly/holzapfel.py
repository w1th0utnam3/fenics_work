import numpy

import dolfin
import dolfin.fem.assembling
from dolfin import MPI

import ufl
from ufl import inner, dot, grad, dx, tr, det, ln

import time


def assemble_test(cell_batch_size: int):
    mesh = dolfin.UnitCubeMesh(MPI.comm_world, 40, 40, 40)

    def isochoric(F):
        C = F.T*F

        I_1 = tr(C)
        I4_f = dot(e_f, C*e_f)
        I4_s = dot(e_s, C*e_s)
        I8_fs = dot(e_f, C*e_s)

        def cutoff(x):
            return 1.0/(1.0 + ufl.exp(-(x - 1.0)*30.0))

        def scaled_exp(a0, a1, argument):
            return a0/(2.0*a1)*(ufl.exp(b*argument) - 1)

        E_1 = scaled_exp(a, b, I_1 - 3.)

        E_f = cutoff(I4_f)*scaled_exp(a_f, b_f, (I4_f - 1.)**2)
        E_s = cutoff(I4_s)*scaled_exp(a_s, b_s, (I4_s - 1.)**2)
        E_3 = scaled_exp(a_fs, b_fs, I8_fs**2)

        E = E_1 + E_f + E_s + E_3
        return E

    cell = mesh.ufl_cell()

    lamda = dolfin.Constant(0.48, cell)
    a = dolfin.Constant(1.0, cell)
    b = dolfin.Constant(1.0, cell)
    a_s = dolfin.Constant(1.0, cell)
    b_s = dolfin.Constant(1.0, cell)
    a_f = dolfin.Constant(1.0, cell)
    b_f = dolfin.Constant(1.0, cell)
    a_fs = dolfin.Constant(1.0, cell)
    b_fs = dolfin.Constant(1.0, cell)

    # For more fun, make these general vector fields rather than
    # constants:
    e_s = dolfin.Constant([0.0, 1.0, 0.0], cell)
    e_f = dolfin.Constant([1.0, 0.0, 0.0], cell)

    V = dolfin.FunctionSpace(mesh, ufl.VectorElement("CG", cell, 1))
    u = dolfin.Function(V)
    du = dolfin.function.argument.TrialFunction(V)
    v = dolfin.function.argument.TestFunction(V)

    # Misc elasticity related tensors and other quantities
    F = grad(u) + ufl.Identity(3)
    F = ufl.variable(F)
    J = det(F)
    Fbar = J**(-1.0/3.0)*F

    # Define energy
    E_volumetric = lamda*0.5*ln(J)**2
    psi = isochoric(Fbar) + E_volumetric

    # Find first Piola-Kircchoff tensor
    P = ufl.diff(psi, F)

    # Define the variational formulation
    F = inner(P, grad(v))*dx

    # Take the derivative
    J = ufl.derivative(F, u, du)

    a, L = J, F

    if cell_batch_size > 1:
        cxx_flags = "-O2 -ftree-vectorize -funroll-loops -march=native -mtune=native"
    else:
        cxx_flags = "-O2"

    assembler = dolfin.fem.assembling.Assembler([[a]], [L], [],
                                                form_compiler_parameters={"cell_batch_size": cell_batch_size,
                                                                          "enable_cross_cell_gcc_ext": True,
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
