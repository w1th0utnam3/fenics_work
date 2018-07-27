import numpy as np
from dolfin import *
from ufl.geometry import JacobianDeterminant
from ufl.classes import ReferenceGrad as Grad
from ufl.classes import ReferenceValue as Val
from dolfin.la import PETScMatrix

from . import utils

def generate_ref_tensor(element: FiniteElement = None):
    def monkey_patch_ufl():
        from ufl.referencevalue import ReferenceValue
        oldinit = ReferenceValue.__init__

        def newinit(self, f):
            if isinstance(f, ReferenceValue):
                f = f.ufl_operands[0]
            oldinit(self, f)

        ReferenceValue.__init__ = newinit

    monkey_patch_ufl()

    def generate_reference_tetrahedron_mesh():
        vertices = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=np.float64)

        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

        mesh = Mesh(MPI.comm_world, CellType.Type.tetrahedron,
                    vertices, cells, [], cpp.mesh.GhostMode.none)

        return mesh

    mesh = generate_reference_tetrahedron_mesh()

    if element is None:
        element = FiniteElement("P", tetrahedron, 1)
    V = FunctionSpace(mesh, element)

    dofmap = V.dofmap().cell_dofs(0)
    dofmap_inverse = np.argsort(dofmap)

    u, v = TrialFunction(V), TestFunction(V)
    detJ = JacobianDeterminant(SpatialCoordinate(mesh))

    A0 = np.zeros((dofmap.size, dofmap.size, mesh.topology.dim, mesh.topology.dim), dtype=np.double)

    for i in range(mesh.topology.dim):
        for j in range(mesh.topology.dim):
            jit_result = jit.jit.ffc_jit(outer(Grad(Val(u)), Grad(Val(v)))[i, j] / detJ * dx)
            ufc_form = cpp.fem.make_ufc_form(jit_result[0])
            a = cpp.fem.Form(ufc_form, [V._cpp_object, V._cpp_object])

            assembler = cpp.fem.Assembler([[a]], [], [])
            A_scp = PETScMatrix()
            assembler.assemble(A_scp, cpp.fem.Assembler.BlockType.monolithic)

            A = utils.scipy2numpy(A_scp)
            A0[:,:,i,j] = A[dofmap_inverse].transpose()

            #print(79 * '=')
            #print("dphi_i/dX({})*dphi_j/dX({})".format(i, j))
            #print(A[dofmap_inverse])

    A0 = A0[dofmap_inverse, :, :, :]
    return A0
