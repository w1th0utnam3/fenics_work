from ufl import *
from typing import Optional


def laplace_form(element, coeff_element: Optional[FiniteElement] = None) -> Form:
    v = TestFunction(element)
    u = TrialFunction(element)

    if coeff_element is None:
        return inner(grad(u), grad(v)) * dx
    else:
        c = Coefficient(coeff_element)
        return c * inner(grad(u), grad(v)) * dx


def laplace_p2tet_coefficient_p1tet():
    return laplace_form(FiniteElement("Lagrange", tetrahedron, 2),
                        FiniteElement("Lagrange", tetrahedron, 1))


def laplace_p1tri():
    return laplace_form(FiniteElement("Lagrange", triangle, 1))


def laplace_p2tet():
    return laplace_form(FiniteElement("Lagrange", tetrahedron, 2))


def biharmonic_p2tet():
    # Elements
    element = FiniteElement("Lagrange", tetrahedron, 2)

    # Trial and test functions
    u = TrialFunction(element)
    v = TestFunction(element)

    # Facet normal, mesh size and right-hand side
    n = FacetNormal(tetrahedron)
    h = 2.0 * Circumradius(tetrahedron)

    # Compute average of mesh size
    h_avg = (h('+') + h('-')) / 2.0

    # Parameters
    alpha = Constant(tetrahedron)

    # Bilinear form
    return inner(div(grad(u)), div(grad(v))) * dx \
           - inner(jump(grad(u), n), avg(div(grad(v)))) * dS \
           - inner(avg(div(grad(u))), jump(grad(v), n)) * dS \
           + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS


def hyperelasticity_p1tet():
    # Coefficient spaces
    element = VectorElement("Lagrange", tetrahedron, 1)

    # Coefficients
    v = TestFunction(element)  # Test function
    du = TrialFunction(element)  # Incremental displacement
    u = Coefficient(element)  # Displacement from previous iteration

    B = Coefficient(element)  # Body force per unit mass
    T = Coefficient(element)  # Traction force on the boundary

    # Kinematics
    d = len(u)
    I = Identity(d)  # Identity tensor
    F = I + grad(u)  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor
    E = (C - I) / 2  # Euler-Lagrange strain tensor
    E = variable(E)

    # Material constants
    mu = Constant(tetrahedron)  # Lame's constants
    lmbda = Constant(tetrahedron)

    # Strain energy function (material model)
    psi = lmbda / 2 * (tr(E) ** 2) + mu * tr(E * E)

    S = diff(psi, E)  # Second Piola-Kirchhoff stress tensor
    P = F * S  # First Piola-Kirchoff stress tensor

    # The variational problem corresponding to hyperelasticity
    L = inner(P, grad(v)) * dx - inner(B, v) * dx - inner(T, v) * ds
    return derivative(L, u, du)
