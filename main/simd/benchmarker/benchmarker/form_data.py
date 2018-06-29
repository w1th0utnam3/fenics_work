import numpy as np
from typing import List

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


def get_form(form_name: str):
    """Lazy construction of the form with the specified name"""
    import benchmarker.forms as forms
    return getattr(forms, form_name)()


def laplace_p2tet_coefficient_p1tet() -> FormTestData:
    form_name = "laplace_p2tet_coefficient_p1tet"

    return FormTestData(
        form_name=form_name,
        form_gen=lambda: get_form(form_name),
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(120 ** 3 / 4) * 4)
    )


def laplace_p1tri() -> FormTestData:
    form_name = "laplace_p1tri"

    return FormTestData(
        form_name=form_name,
        form_gen=lambda: get_form(form_name),
        element_tensor_size=9,
        coefficients=W3,
        coord_dofs=DOF_3x2,
        n_elems=int(np.floor(600 ** 3 / 4) * 4)
    )


def laplace_p2tet() -> FormTestData:
    form_name = "laplace_p2tet"

    return FormTestData(
        form_name=form_name,
        form_gen=lambda: get_form(form_name),
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def biharmonic_p2tet() -> FormTestData:
    form_name = "biharmonic_p2tet"

    return FormTestData(
        form_name=form_name,
        form_gen=lambda: get_form(form_name),
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def hyperelasticity_p1tet() -> FormTestData:
    form_name = "hyperelasticity_p1tet"

    return FormTestData(
        form_name=form_name,
        form_gen=lambda: get_form(form_name),
        element_tensor_size=144,
        coefficients=W3x12,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(140 ** 3 / 4) * 4)
    )


def stokes_p2p1tet() -> FormTestData:
    form_name = "stokes_p2p1tet"

    return FormTestData(
        form_name=form_name,
        form_gen=lambda: get_form(form_name),
        element_tensor_size=1156,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(110 ** 3 / 4) * 4)
    )


def get_all_forms() -> List[FormTestData]:
    return [
        laplace_p1tri(),
        laplace_p2tet(),
        laplace_p2tet_coefficient_p1tet(),
        biharmonic_p2tet(),
        hyperelasticity_p1tet(),
        stokes_p2p1tet()
    ]
