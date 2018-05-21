# Notes

## Example code

```
def poisson():
    # Create mesh and define function space
    mesh = RectangleMesh.create(MPI.comm_world,
                                [Point(0, 0), Point(1, 1)], [32, 32],
                                CellType.Type.triangle, dolfin.cpp.mesh.GhostMode.none)
    V = FunctionSpace(mesh, "Lagrange", 1)

    cmap = dolfin.fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return np.logical_or(x[:, 0] < DOLFIN_EPS, x[:, 0] > 1.0 - DOLFIN_EPS)

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("sin(5*x[0])", degree=2)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    # Compute solution
    u = Function(V)
    form = a == L
    solve(form, u, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, form_compiler_parameters=fcp)

    # Save solution in XDMF format
    #file = XDMFFile(MPI.comm_world, "poisson.xdmf")
    #file.write(u, XDMFFile.Encoding.HDF5)

    return
```

## Trace of solve call

**User code:**
 - User makes call `solve(form, u, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, form_compiler_parameters=fcp)`

**Dolfin:**
 - `solve` method is in `dolfinx/python/dolfin/fem/solving.py`
 - `_solve_varproblem()`
 - `a = Form(eq.lhs, ...)` with `Form` constructor in `dolfinx/python/dolfin/fem/form.py`

**FFC:**
 - `ufc_form = ffc_jit(...)` which calls `ffc.jit()` in `ffcx/ffc/jitcompiler.py`
 - `build()`

**Dijitso:**
 - `dijitso.jit()` in `dijitso/dijitso/jit.py`, gets `generate` from `ffcx/ffc/jitcompiler.py` as parameter

**FFC:** (traced only for the bilinear form)
 - `generate()` in `ffcx/ffc/jitcompiler.py`
 - `compile_form()` in `ffcx/ffc/compiler.py`
 - `compile_ufl_objects(forms, "form",...)` in `ffcx/ffc/compiler.py`
    1. `analyze_ufl_objects()` in `ffcx/ffc/analysis.py`, already generates integral:
		```
		weight * |(J[0, 0] * J[1, 1] + -1 * J[0, 1] * J[1, 0])| * (sum_{i_8} ({ A | A_{i_9} = sum_{i_{10}} ({ A | A_{i_{13}, i_{14}} = ([
		[J[1, 1], -1 * J[0, 1]],
		[-1 * J[1, 0], J[0, 0]]
		])[i_{13}, i_{14}] / (J[0, 0] * J[1, 1] + -1 * J[0, 1] * J[1, 0]) })[i_{10}, i_9] * (reference_grad(reference_value(v_0)))[i_{10}]  })[i_8] * ({ A | A_{i_{11}} = sum_{i_{12}} ({ A | A_{i_{13}, i_{14}} = ([
		[J[1, 1], -1 * J[0, 1]],
		[-1 * J[1, 0], J[0, 0]]
		])[i_{13}, i_{14}] / (J[0, 0] * J[1, 1] + -1 * J[0, 1] * J[1, 0]) })[i_{12}, i_{11}] * (reference_grad(reference_value(v_1)))[i_{12}]  })[i_8] )
		```
		and corresponding UFL operator "AST"
	2. `compute_ir()` in `ffcx/ffc/representation.py`
	3. `optimize_ir()` in `ffcx/ffc/optimization.py`
	4. `generate_code()` in `ffcx/ffc/codegeneration.py`
		- `ufc_integral_generator()` in `ffcx/ffc/backends/ufc/integrals.py` calls `r = pick_representation()` from `ffcx/ffc/representation.py` followed by `r.generate_integral_code(ir,...)`
			- Usually `generate_integral_code()` from `ffcx/ffc/uflacs/uflacsgenerator.py`
			- `IntegralGenerator.generate()` to generate code AST for `tabulate_tensor`
			- `generate()` in `ffcx/ffc/uflacs/integralgenerator.py`
			- Calls several AST generation functions, for example `generate_preintegrated_dofblock_partition()`
	    - `ufc_form_generator()` in `ffcx/ffc/backends/ufc/form.py`
	5. `format_code()` in `ffcx/ffc/formatting.py`, this generates the final source and header files for compilation
