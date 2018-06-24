import ufl
from ufl import *

import numpy as np
from typing import List, Optional

from benchmarker.types import FormTestData


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
        coefficients=np.asarray([
            [1.0, 1.1, 1.2, 1.3]
        ], dtype=np.double),
        coord_dofs=np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.double),
        n_elems=int(np.floor(120 ** 3 / 4) * 4)
    )


def laplace_p1tri() -> FormTestData:
    def a():
        return laplace_form(FiniteElement("Lagrange", triangle, 1))

    return FormTestData(
        form_name="laplace_p1tri",
        code_gen=a,
        element_tensor_size=9,
        coefficients=np.asarray([
            [1.0, 1.1, 1.2]
        ], dtype=np.double),
        coord_dofs=np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=np.double),
        n_elems=int(np.floor(600 ** 3 / 4) * 4)
    )


def laplace_p2tet() -> FormTestData:
    def a():
        return laplace_form(FiniteElement("Lagrange", tetrahedron, 2))

    return FormTestData(
        form_name="laplace_p2tet",
        code_gen=a,
        element_tensor_size=100,
        coefficients=np.asarray([
            [1.0, 1.1, 1.2, 1.3]
        ], dtype=np.double),
        coord_dofs=np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.double),
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
        coefficients=np.asarray([
            [1.0, 1.1, 1.2, 1.3]
        ], dtype=np.double),
        coord_dofs=np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.double),
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )

def get_all_forms() -> List[FormTestData]:
    return [
        laplace_p1tri(),
        laplace_p2tet(),
        laplace_p2tet_coefficient_p1tet(),
        biharmonic_p2tet()
    ]
