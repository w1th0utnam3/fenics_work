import ufl
from ufl import *

import numpy as np
from typing import List, Optional

from benchmarker.types import FormTestData

W3 = np.linspace(1, 2, num=3, dtype=np.double)[np.newaxis,:]
W4 = np.linspace(1, 2, num=4, dtype=np.double)[np.newaxis,:]
W12 = np.linspace(1, 2, num=12, dtype=np.double)[np.newaxis,:]
W3x12 = np.repeat(W12, 3, axis=0)

DOF_3x2 = np.asarray([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
], dtype=np.double)

DOF_4x3 = np.asarray([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.double)


def laplace_form(element, coeff_element: Optional[ufl.FiniteElement] = None) -> ufl.Form:
    v = TestFunction(element)
    u = TrialFunction(element)

    if coeff_element is None:
        return inner(grad(u), grad(v)) * dx
    else:
        c = Coefficient(coeff_element)
        return c * inner(grad(u), grad(v)) * dx


def laplace_p2tet_coefficient_p1tet() -> FormTestData:
    def a():
        return laplace_form(FiniteElement("Lagrange", tetrahedron, 2),
                            FiniteElement("Lagrange", tetrahedron, 1))

    return FormTestData(
        form_name="laplace_p2tet_coefficient_p1tet",
        code_gen=a,
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(120 ** 3 / 4) * 4)
    )


def laplace_p1tri() -> FormTestData:
    def a():
        return laplace_form(FiniteElement("Lagrange", triangle, 1))

    return FormTestData(
        form_name="laplace_p1tri",
        code_gen=a,
        element_tensor_size=9,
        coefficients=W3,
        coord_dofs=DOF_3x2,
        n_elems=int(np.floor(600 ** 3 / 4) * 4)
    )


def laplace_p2tet() -> FormTestData:
    def a():
        return laplace_form(FiniteElement("Lagrange", tetrahedron, 2))

    return FormTestData(
        form_name="laplace_p2tet",
        code_gen=a,
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def biharmonic_p2tet() -> FormTestData:
    def a():
        # Elements
        element = FiniteElement("Lagrange", tetrahedron, 2)

        # Trial and test functions
        u = TrialFunction(element)
        v = TestFunction(element)

        # Facet normal, mesh size and right-hand side
        n = FacetNormal(tetrahedron)
        h = 2.0 * Circumradius(tetrahedron)
        f = Coefficient(element)

        # Compute average of mesh size
        h_avg = (h('+') + h('-')) / 2.0

        # Parameters
        alpha = Constant(tetrahedron)

        # Bilinear form
        return inner(div(grad(u)), div(grad(v))) * dx \
               - inner(jump(grad(u), n), avg(div(grad(v)))) * dS \
               - inner(avg(div(grad(u))), jump(grad(v), n)) * dS \
               + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS

    return FormTestData(
        form_name="biharmonic_p2tet",
        code_gen=a,
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def hyperelasticity_p1tet() -> FormTestData:
    def a():
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

    return FormTestData(
        form_name="hyperelasticity_p1tet",
        code_gen=a,
        element_tensor_size=144,
        coefficients=W3x12,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(140 ** 3 / 4) * 4)
    )


def get_all_forms() -> List[FormTestData]:
    return [
        laplace_p1tri(),
        laplace_p2tet(),
        laplace_p2tet_coefficient_p1tet(),
        biharmonic_p2tet(),
        hyperelasticity_p1tet()
    ]
