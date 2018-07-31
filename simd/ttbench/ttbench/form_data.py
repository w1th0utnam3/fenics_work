import inspect
import numpy as np
from typing import List, Tuple

from ttbench.types import FormTestData

W1 = np.linspace(1, 2, num=1, dtype=np.double)[np.newaxis, :]
W3 = np.linspace(1, 2, num=3, dtype=np.double)[np.newaxis, :]
W4 = np.linspace(1, 2, num=4, dtype=np.double)[np.newaxis, :]
W12 = np.linspace(1, 2, num=12, dtype=np.double)[np.newaxis, :]
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


def get_form_code(form_name: str) -> Tuple[str, str]:
    import ttbench.forms
    form_env = inspect.getsource(ttbench.forms)
    form_expr = "{}()".format(form_name)
    return form_expr, form_env


def laplace_p2tet_coefficient_p1tet() -> FormTestData:
    form_name = "laplace_p2tet_coefficient_p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(110 ** 3 / 4) * 4)
    )


def laplace_p1tri() -> FormTestData:
    form_name = "laplace_p1tri"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=9,
        coefficients=W3,
        coord_dofs=DOF_3x2,
        n_elems=int(np.floor(500 ** 3 / 4) * 4)
    )


def laplace_p2tet() -> FormTestData:
    form_name = "laplace_p2tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def biharmonic_p2tet() -> FormTestData:
    form_name = "biharmonic_p2tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=100,
        coefficients=W4,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def hyperelasticity_energy_p2tet() -> FormTestData:
    form_name = "hyperelasticity_energy_p2tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=1,
        coefficients=W3x12,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(160 ** 3 / 4) * 4)
    )


def hyperelasticity_p1tet() -> FormTestData:
    form_name = "hyperelasticity_p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=144,
        coefficients=W3x12,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(160 ** 3 / 4) * 4)
    )


def stokes_p2p1tet() -> FormTestData:
    form_name = "stokes_p2p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=1156,
        coefficients=W1,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(120 ** 3 / 4) * 4)
    )


def nearly_incompressible_stokes_p2p1tet() -> FormTestData:
    form_name = "nearly_incompressible_stokes_p2p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=1156,
        coefficients=W1,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(120 ** 3 / 4) * 4)
    )


def curlcurl_nedelec3tet() -> FormTestData:
    form_name = "curlcurl_nedelec3tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=2025,
        coefficients=W1,
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(30 ** 3 / 4) * 4)
    )


def get_all_forms() -> List[FormTestData]:
    return [
        laplace_p1tri(),
        laplace_p2tet(),
        laplace_p2tet_coefficient_p1tet(),
        biharmonic_p2tet(),
        hyperelasticity_p1tet(),
        hyperelasticity_energy_p2tet(),
        stokes_p2p1tet(),
        nearly_incompressible_stokes_p2p1tet(),
        # curlcurl_nedelec3tet()
    ]
