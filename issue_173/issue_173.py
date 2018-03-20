from dolfin import *

import sys
import logging
import dijitso; dijitso.set_log_level("debug")

def main():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    fcp = dict(parameters['form_compiler'])

    # Works:
    #fcp['enable_preintegration'] = False

    # Works:
    #fcp['representation'] = 'tsfc'
    #fcp['mode'] = 'spectral'

    # Original issue code is fixed:
    """
    mesh = UnitCubeMesh(1, 1, 1)
    V = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 3)
    u, v = TrialFunction(V), TestFunction(V)
    a = (inner(curl(u), curl(v)) - Constant(1)*inner(u, v))*dx
    assemble(a, form_compiler_parameters=fcp)
    """

    # Now onto regression...:
    mesh = UnitSquareMesh(1, 1)

    # Elements
    element = FunctionSpace(mesh, "Lagrange", 2)

    # Trial and test functions
    u = TrialFunction(element)
    v = TestFunction(element)

    # Facet normal, mesh size and right-hand side
    n = FacetNormal(mesh)
    h = 2.0 * Circumradius(mesh)

    # Compute average of mesh size
    h_avg = (h('+') + h('-')) / 2.0

    # Parameters
    alpha = Constant(1)

    # Bilinear form
    a = inner(div(grad(u)), div(grad(v))) * dx \
        - inner(jump(grad(u), n), avg(div(grad(v)))) * dS \
        - inner(avg(div(grad(u))), jump(grad(v), n)) * dS \
        + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS

    assemble(a, form_compiler_parameters=fcp)


if __name__ == '__main__':
    main()