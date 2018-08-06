import numpy as np

import dolfin
from dolfin import DOLFIN_EPS

import ufl
from ufl import inner, grad, dx
from ufl import dot, ds, tr, det, ln


def laplace_forms(mesh, el):
    Q = dolfin.function.functionspace.FunctionSpace(mesh, el)

    u = dolfin.function.argument.TrialFunction(Q)
    v = dolfin.function.argument.TestFunction(Q)

    def boundary(x):
        return np.sum(np.logical_or(x < DOLFIN_EPS, x > 1.0 - DOLFIN_EPS), axis=1) > 0

    u_bc = dolfin.function.constant.Constant(50.0)
    bc = dolfin.fem.dirichletbc.DirichletBC(Q, u_bc, boundary)

    f = dolfin.function.expression.Expression("0.4*x[1]*x[2]", element=el)

    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx

    return a, L, bc


def laplace_coeff_forms(mesh, el, coeff_el):
    Q = dolfin.function.functionspace.FunctionSpace(mesh, el)

    u = dolfin.function.argument.TrialFunction(Q)
    v = dolfin.function.argument.TestFunction(Q)

    def boundary(x):
        return np.sum(np.logical_or(x < DOLFIN_EPS, x > 1.0 - DOLFIN_EPS), axis=1) > 0

    u_bc = dolfin.function.constant.Constant(50.0)
    bc = dolfin.fem.dirichletbc.DirichletBC(Q, u_bc, boundary)

    c = dolfin.function.expression.Expression("3.14*x[0]", element=coeff_el)
    f = dolfin.function.expression.Expression("0.4*x[1]*x[2]", element=el)

    a = inner(c * grad(u), grad(v)) * dx
    L = f * v * dx

    return a, L, bc


def hyperelasticity_forms(mesh, vec_el):
    cell = mesh.ufl_cell()

    Q = dolfin.FunctionSpace(mesh, vec_el)

    # Coefficients
    v = dolfin.function.argument.TestFunction(Q)  # Test function
    du = dolfin.function.argument.TrialFunction(Q)  # Incremental displacement
    u = dolfin.Function(Q)  # Displacement from previous iteration

    B = dolfin.Constant((0.0, -0.5, 0.0), cell)  # Body force per unit volume
    T = dolfin.Constant((0.1, 0.0, 0.0), cell)  # Traction force on the boundary

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

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu / 2) * (Ic - 3) - mu * ln(J) + (lmbda / 2) * (ln(J)) ** 2

    # Total potential energy
    Pi = psi * dx - dot(B, u) * dx - dot(T, u) * ds

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = ufl.derivative(Pi, u, v)

    # Compute Jacobian of F
    J = ufl.derivative(F, u, du)

    return J, F, None
