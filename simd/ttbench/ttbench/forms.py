from ufl import *

def laplace_form(element: FiniteElement, coeff_element: FiniteElement = None) -> Form:
    v = TestFunction(element)
    u = TrialFunction(element)

    if coeff_element is None:
        return inner(grad(u), grad(v)) * dx
    else:
        c = Coefficient(coeff_element)
        return c * inner(grad(u), grad(v)) * dx


def laplace_p2tet_coefficient_p1tet() -> Form:
    return laplace_form(FiniteElement("Lagrange", tetrahedron, 2),
                        FiniteElement("Lagrange", tetrahedron, 1))


def laplace_p1tri() -> Form:
    return laplace_form(FiniteElement("Lagrange", triangle, 1))


def laplace_p2tet() -> Form:
    return laplace_form(FiniteElement("Lagrange", tetrahedron, 2))


def biharmonic_form(element: FiniteElement) -> Form:
    cell = element.cell()

    # Trial and test functions
    u = TrialFunction(element)
    v = TestFunction(element)

    # Facet normal, mesh size and right-hand side
    n = FacetNormal(cell)
    h = 2.0 * Circumradius(cell)

    # Compute average of mesh size
    h_avg = (h('+') + h('-')) / 2.0

    # Parameters
    alpha = Constant(cell)

    # Bilinear form
    return inner(div(grad(u)), div(grad(v))) * dx \
           - inner(jump(grad(u), n), avg(div(grad(v)))) * dS \
           - inner(avg(div(grad(u))), jump(grad(v), n)) * dS \
           + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS


def biharmonic_p2tet() -> Form:
    element = FiniteElement("Lagrange", tetrahedron, 2)
    return biharmonic_form(element)


def hyperelasticity_form(element: VectorElement):
    cell = element.cell()

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
    mu = Constant(cell)  # Lame's constants
    lmbda = Constant(cell)

    # Strain energy function (material model)
    psi = lmbda / 2 * (tr(E) ** 2) + mu * tr(E * E)
    E_strain = psi*dx

    S = diff(psi, E)  # Second Piola-Kirchhoff stress tensor
    P = F * S  # First Piola-Kirchoff stress tensor

    # The variational problem corresponding to hyperelasticity
    L = inner(P, grad(v)) * dx - inner(B, v) * dx - inner(T, v) * ds
    return E_strain, L, derivative(L, u, du)


def hyperelasticity_energy_p2tet() -> Form:
    element = VectorElement("Lagrange", tetrahedron, 2)
    E_strain, L, a = hyperelasticity_form(element)
    return E_strain


def hyperelasticity_p1tet() -> Form:
    element = VectorElement("Lagrange", tetrahedron, 1)
    E_strain, L, a = hyperelasticity_form(element)
    return a


def stokes_form(vector: VectorElement, scalar: FiniteElement) -> Form:
    system = vector * scalar

    (u, p) = TrialFunctions(system)
    (v, q) = TestFunctions(system)

    f = Coefficient(vector)

    a =  (inner(grad(u), grad(v)) - div(v)*p + div(u)*q)*dx
    L = dot(f, v) * dx

    return a


def stokes_p2p1tet() -> Form:
    cell = tetrahedron
    vector = VectorElement("Lagrange", cell, 2)
    scalar = FiniteElement("Lagrange", cell, 1)
    return stokes_form(vector, scalar)


def nearly_incompressible_stokes_form(vector: VectorElement, scalar: FiniteElement) -> Form:
    system = vector * scalar
    cell = system.cell()

    (u, p) = TrialFunctions(system)
    (v, q) = TestFunctions(system)

    f = Coefficient(vector)
    delta = Constant(cell)

    a = (inner(grad(u), grad(v)) + p*div(v) + q*div(u) - delta*p*q)*dx
    L = dot(f, v) * dx

    return a


def nearly_incompressible_stokes_p2p1tet() -> Form:
    cell = tetrahedron
    vector = VectorElement("Lagrange", cell, 2)
    scalar = FiniteElement("Lagrange", cell, 1)
    return nearly_incompressible_stokes_form(vector, scalar)


def curlcurl_form(element: FiniteElement) -> Form:
    u = TrialFunction(element)
    v = TestFunction(element)

    a = (inner(curl(u), curl(v)) - inner(u, v)) * dx

    return a


def curlcurl_nedelec3tet() -> Form:
    element = FiniteElement("Nedelec 1st kind H(curl)", tetrahedron, 3)
    return curlcurl_form(element)
