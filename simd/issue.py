import ufl
from ufl import inner, dx

from minidolfin.assembling import jit_compile_form


# for fixing https://github.com/FEniCS/ffcx/pull/25

cell = ufl.tetrahedron
element = ufl.VectorElement("Nedelec 1st kind H(curl)", cell, 3)

u = ufl.TrialFunction(element)
v = ufl.TestFunction(element)

a = inner(u, v)*dx

jit_compile_form(a, {"compiler": "ffc",
                     "max_preintegrated_unrolled_table_size": 2000})
