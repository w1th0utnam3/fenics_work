import inspect
import numpy as np
from typing import List, Tuple

from ttbench.types import FormTestData

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


def w(i, j):
    vals = np.linspace(1, 2, num=i, dtype=np.double)[np.newaxis, :]
    return np.repeat(vals, j, axis=0)


def get_form_code(form_name: str) -> Tuple[str, str]:
    import ttbench.forms_ufl
    form_env = inspect.getsource(ttbench.forms_ufl)
    form_expr = "{}()".format(form_name)
    return form_expr, form_env


def laplace_p1tri() -> FormTestData:
    form_name = "laplace_p1tri"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=9,
        coefficients=w(3, 1),
        coord_dofs=DOF_3x2,
        n_elems=int(np.floor(500 ** 3 / 4) * 4)
    )


def laplace_p2tet() -> FormTestData:
    form_name = "laplace_p2tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=100,
        coefficients=w(4, 1),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def laplace_p2tet_action() -> FormTestData:
    form_name = "laplace_p2tet_action"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=10,
        coefficients=w(1, 10),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def biharmonic_p2tet() -> FormTestData:
    form_name = "biharmonic_p2tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=100,
        coefficients=w(1, 1),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(240 ** 3 / 4) * 4)
    )


def laplace_p2tet_coefficient_p1tet() -> FormTestData:
    form_name = "laplace_p2tet_coefficient_p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=100,
        coefficients=w(1, 4),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(110 ** 3 / 4) * 4)
    )


def laplace_p2tet_coefficient_p1tet_action() -> FormTestData:
    form_name = "laplace_p2tet_coefficient_p1tet_action"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=10,
        coefficients=w(2, 10),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(110 ** 3 / 4) * 4)
    )


def hyperelasticity_energy_p2tet() -> FormTestData:
    form_name = "hyperelasticity_energy_p2tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=1,
        coefficients=w(3, 30),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(160 ** 3 / 4) * 4)
    )


def hyperelasticity_p1tet() -> FormTestData:
    form_name = "hyperelasticity_p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=144,
        coefficients=w(5, 12),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(160 ** 3 / 4) * 4)
    )


def hyperelasticity_action_p1tet() -> FormTestData:
    form_name = "hyperelasticity_action_p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=12,
        coefficients=w(4, 12),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(160 ** 3 / 4) * 4)
    )


def holzapfel_p1tet() -> FormTestData:
    form_name = "holzapfel_p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=144,
        coefficients=w(12, 12),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(70 ** 3 / 4) * 4)
    )


def holzapfel_action_p1tet() -> FormTestData:
    form_name = "holzapfel_action_p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=12,
        coefficients=w(13, 12),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(70 ** 3 / 4) * 4)
    )


def stokes_p2p1tet() -> FormTestData:
    form_name = "stokes_p2p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=1156,
        coefficients=w(1, 1),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(120 ** 3 / 4) * 4)
    )


def stokes_action_p2p1tet() -> FormTestData:
    form_name = "stokes_action_p2p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=34,
        coefficients=w(1, 34),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(120 ** 3 / 4) * 4)
    )


def nearly_incompressible_stokes_p2p1tet() -> FormTestData:
    form_name = "nearly_incompressible_stokes_p2p1tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=1156,
        coefficients=w(1, 1),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(120 ** 3 / 4) * 4)
    )


def curlcurl_nedelec3tet() -> FormTestData:
    form_name = "curlcurl_nedelec3tet"

    return FormTestData(
        form_name=form_name,
        form_code=get_form_code(form_name),
        element_tensor_size=2025,
        coefficients=w(1, 1),
        coord_dofs=DOF_4x3,
        n_elems=int(np.floor(30 ** 3 / 4) * 4)
    )


def get_all_forms() -> List[FormTestData]:
    return [
        laplace_p1tri(),
        laplace_p2tet(),
        laplace_p2tet_action(),
        laplace_p2tet_coefficient_p1tet(),
        laplace_p2tet_coefficient_p1tet_action(),
        biharmonic_p2tet(),
        hyperelasticity_p1tet(),
        hyperelasticity_energy_p2tet(),
        hyperelasticity_action_p1tet(),
        # holzapfel_p1tet(),
        # holzapfel_action_p1tet(),
        stokes_p2p1tet(),
        stokes_action_p2p1tet(),
        nearly_incompressible_stokes_p2p1tet(),
        # curlcurl_nedelec3tet()
    ]


def get_bilinear_forms() -> List[FormTestData]:
    return [
        laplace_p1tri(),
        laplace_p2tet(),
        laplace_p2tet_coefficient_p1tet(),
        biharmonic_p2tet(),
        hyperelasticity_p1tet(),
        stokes_p2p1tet(),
        nearly_incompressible_stokes_p2p1tet(),
        # holzapfel_p1tet(),
        # curlcurl_nedelec3tet()
    ]


def get_linear_forms() -> List[FormTestData]:
    return [
        laplace_p2tet_action(),
        laplace_p2tet_coefficient_p1tet_action(),
        hyperelasticity_action_p1tet(),
        stokes_action_p2p1tet(),
        # holzapfel_action_p1tet(),
    ]
