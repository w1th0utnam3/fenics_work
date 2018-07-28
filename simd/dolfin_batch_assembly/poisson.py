import numpy

import dolfin
from dolfin import DOLFIN_EPS, MPI

import ufl
from ufl import inner, grad, dx


def assemble_test(cell_batch_size: int):
        mesh = dolfin.UnitCubeMesh(MPI.comm_world, 2, 3, 4)
        element_p1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        element_p2 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)

        Q = dolfin.function.functionspace.FunctionSpace(mesh, element_p2)

        u = dolfin.function.argument.TrialFunction(Q)
        v = dolfin.function.argument.TestFunction(Q)

        def boundary(x):
            return numpy.sum(numpy.logical_or(x < DOLFIN_EPS, x > 1.0 - DOLFIN_EPS), axis=1) > 0

        u_bc = dolfin.function.constant.Constant(50.0)
        bc = dolfin.fem.dirichletbc.DirichletBC(Q, u_bc, boundary)

        c = dolfin.function.expression.Expression("3.14*x[0]", element=element_p1)
        f = dolfin.function.expression.Expression("0.4*x[1]*x[2]", element=element_p2)

        a = inner(c * grad(u), grad(v)) * dx
        L = f * v * dx

        if cell_batch_size > 1:
            cxx_flags = "-O2 -ftree-vectorize -funroll-loops -march=native -mtune=native"
        else:
            cxx_flags = "-O2"

        # Create assembler
        assembler = dolfin.fem.assembling.Assembler([[a]], [L], [bc],
                                                    form_compiler_parameters={"cell_batch_size": cell_batch_size,
                                                                              "enable_cross_element_gcc_ext": True,
                                                                              "cpp_optimize_flags": cxx_flags})

        A, b = assembler.assemble(mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)

        return A, b


A1, b1 = assemble_test(cell_batch_size=1)
A4, b4 = assemble_test(cell_batch_size=4)

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
