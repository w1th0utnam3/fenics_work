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

	fcp = parameters['form_compiler']
	#fcp.add('enable_preintegration', False)  # Fixes the problem

	mesh = UnitCubeMesh(1, 1, 1)
	V = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 3)
	u, v = TrialFunction(V), TestFunction(V)
	a = (inner(curl(u), curl(v)) - Constant(1)*inner(u, v))*dx
	assemble(a)

if __name__ == '__main__':
	main()