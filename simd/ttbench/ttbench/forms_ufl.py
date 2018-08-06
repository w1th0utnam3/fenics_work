from ufl import *

def laplace_form(element: FiniteElement, coeff_element: FiniteElement = None) -> Form:
    v = TestFunction(element)
    u = TrialFunction(element)

    if coeff_element is None:
        return inner(grad(u), grad(v)) * dx
    else:
        c = Coefficient(coeff_element)
        return inner(c * grad(u), grad(v)) * dx


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


def holzapfel_form(vector: VectorElement, scalar: FiniteElement) -> Form:
    cell = vector.cell()

    v = TestFunction(vector)  # Test function
    du = TrialFunction(vector)  # Incremental displacement
    u = Coefficient(vector)  # Displacement from previous iteration

    def isochoric(F):
        C = F.T * F

        I_1 = tr(C)
        I4_f = dot(e_f, C * e_f)
        I4_s = dot(e_s, C * e_s)
        I8_fs = dot(e_f, C * e_s)

        def cutoff(x):
            return 1.0 / (1.0 + exp(-(x - 1.0) * 30.0))

        def scaled_exp(a0, a1, argument):
            return a0 / (2.0 * a1) * (exp(b * argument) - 1)

        E_1 = scaled_exp(a, b, I_1 - 3.)

        E_f = cutoff(I4_f) * scaled_exp(a_f, b_f, (I4_f - 1.) ** 2)
        E_s = cutoff(I4_s) * scaled_exp(a_s, b_s, (I4_s - 1.) ** 2)
        E_3 = scaled_exp(a_fs, b_fs, I8_fs ** 2)

        E = E_1 + E_f + E_s + E_3
        return E

    lmbda = Constant(cell)
    a = Constant(cell)
    b = Constant(cell)
    a_s = Constant(cell)
    b_s = Constant(cell)
    a_f = Constant(cell)
    b_f = Constant(cell)
    a_fs = Constant(cell)
    b_fs = Constant(cell)

    # For more fun, make these general vector fields rather than
    # constants:
    e_s = VectorConstant(cell)
    e_f = VectorConstant(cell)

    # Misc elasticity related tensors and other quantities
    F = grad(u) + Identity(3)
    F = variable(F)
    J = det(F)
    Fbar = J ** (-1.0 / 3.0) * F

    # Define energy
    E_volumetric = lmbda * 0.5 * ln(J) ** 2
    psi = isochoric(Fbar) + E_volumetric

    # Find first Piola-Kircchoff tensor
    P = diff(psi, F)

    # Define the variational formulation
    F = inner(P, grad(v)) * dx

    # Take the derivative
    J = derivative(F, u, du)

    return J


def holzapfel_p1tet():
    cell = tetrahedron
    vector = VectorElement("Lagrange", cell, 1)
    scalar = FiniteElement("Lagrange", cell, 1)
    return holzapfel_form(vector, scalar)


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
