import os
import sys
import pip
import argparse
import subprocess

REPOS = [
		"https://bitbucket.org/fenics-project/fiat.git",
		"https://bitbucket.org/fenics-project/ufl.git",
		"https://bitbucket.org/fenics-project/dijitso.git",
		"https://bitbucket.org/fenics-project/ffc.git",
		"https://github.com/blechta/tsfc.git",
		"https://github.com/blechta/COFFEE.git",
		"https://github.com/blechta/FInAT.git",
		"https://bitbucket.org/fenics-project/dolfin.git",
		"https://github.com/pybind/pybind11.git"
	]

PIP_INSTALLS = [
	"fiat",
	"ufl",
	"dijitso",
	"ffc",
	"tsfc",
	"COFFEE",
	"FInAT"
]

def print_stdout(args, raise_on_nonzero=False, **kwargs):
	sys.stdout.flush()
	kwargs["bufsize"] = 1
	kwargs["stdout"] = subprocess.PIPE
	#kwargs["stderr"] = subprocess.STDOUT
	process = subprocess.Popen(args, **kwargs)
	while True:
		output = process.stdout.readline()
		if len(output.strip()) == 0 and process.poll() is not None:
			break
		if output:
			print(output.strip().decode())
			sys.stdout.flush()

	rc = process.poll()
	sys.stdout.flush()

	if raise_on_nonzero and rc != 0:
		raise RuntimeError(f"Subprocess {args[0]} did not terminate successfully.")

	return process.poll()

def clone_repos(src_directory: str):
	print("Starting to clone all git repositories...")
	print("")
	for repo in REPOS:
		print_stdout(["git", "clone", repo], cwd=src_directory)
		print("")

	return 0

def pip_install(src_dir: str):
	print("Insalling python packages from repositories...")
	for pkg in PIP_INSTALLS:
		pkg_path = os.path.join(src_dir, pkg)
		pip.main(["install", "-e", f"{pkg_path}"])
		print("")

	for pkg in ["six", "singledispatch", "pulp", "pytest", "pybind11"]:
		pip.main(["install", pkg])
		print("")

	return 0

def pybind_build(src_dir: str, build_dir: str, pybind_dir: str):
	pybind_build_path = os.path.join(build_dir, "pybind11")
	os.makedirs(pybind_build_path, exist_ok=True)

	print_stdout(["cmake", "-DPYBIND11_TEST=off", 
		f"-DCMAKE_INSTALL_PREFIX={pybind_dir}",
		f"{src_dir}/pybind11"],
		cwd=pybind_build_path)
	print("")

	make_return = print_stdout(["make", "install"], cwd=pybind_build_path)
	print("")

	if make_return != 0:
		print("pybind make install did not return successfully!")
		return 1

	return 0

def dolfin_build(src_dir: str, build_dir: str, venv_dir: str, dolfin_dir: str, pybind_dir: str, jobs: int):
	dolfin_build_path = os.path.join(build_dir, "dolfin")
	os.makedirs(dolfin_build_path, exist_ok=True)
	
	try:
		print("Running CMake...")
		activate_file = os.path.join(venv_dir, "bin", "activate")
		print_stdout([f". {activate_file} && cmake -DCMAKE_INSTALL_PREFIX={dolfin_dir} {src_dir}/dolfin"],
			raise_on_nonzero=True,
			cwd=dolfin_build_path,
			shell=True
		)
		print("DOLFIN CMake was successful.")

		print("Running make...")
		print_stdout(["make", f"-j{jobs}"], raise_on_nonzero=True, cwd=dolfin_build_path)
		print("DOLFIN make was successful.")
		print("Running make install...")
		print_stdout(["make", "install"], raise_on_nonzero=True, cwd=dolfin_build_path)
		print("DOLFIN make install was successful.")

		print("")
		print("Installing DOLFIN Python package...")
		environment = os.environ.copy()
		environment["PYBIND11_DIR"] = pybind_dir
		environment["DOLFIN_DIR"] = dolfin_dir
		print_stdout([os.path.join(venv_dir, "bin", "pip3"), "install", "-e", "."], 
			raise_on_nonzero=True, 
			cwd=os.path.join(src_dir, "dolfin", "python"),
			env=environment
		)
		print("Installing DOLFIN Python was successful.")

	except RuntimeError as err:
		print(err)
		return 1

	return 0

def main():
	parser = argparse.ArgumentParser(description="install and set up an environment for FEniCS from git")
	parser.add_argument("-r", "--repo-dir", type=str, help="Directory where source directories (repositories) will be stored/are already located.")
	parser.add_argument("-y", "--yes", action="store_true", help="Respond with 'yes' to all confirmation messages.")
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-l", "--local", action="store_true", help="Don't clone any repositories, use local files.")
	group.add_argument("-co", "--clone-only", action="store_true", help="Only clone the required repositories, no install/build.")
	parser.add_argument("--internal-stage", type=int, help=argparse.SUPPRESS)
	parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="Number of make jobs to issue for building DOLFIN ('-j' parameter for make). Default is to use 'os.cpu_count()'.")
	parser.add_argument("install_prefix", type=str, help="FEniCS will be installed into this folder. If not specified, the current folder will be used.", nargs='?')

	args = parser.parse_args()

	if args.internal_stage is None:
		import click

	FENICS_DIR = args.install_prefix
	SRC_DIR = args.repo_dir
	BUILD_DIR = None

	if FENICS_DIR is None:
		FENICS_DIR = os.getcwd()

	if SRC_DIR is None:
		SRC_DIR = os.path.join(FENICS_DIR, "src")

	VENV_DIR = os.path.join(FENICS_DIR, "fenics_env")
	BUILD_DIR = os.path.join(FENICS_DIR, "build")
	PYBIND_DIR = os.path.join(FENICS_DIR, "include", "pybind11")
	DOLFIN_DIR = os.path.join(FENICS_DIR, "dolfin")

	# The clone only branch
	if args.clone_only is True:
		if not args.yes:
			try:
				click.confirm(f"Continue cloning repositories into {SRC_DIR}?", abort=True)
			except click.exceptions.Abort:
				return

		print("Only cloning repositories...")
		print()

		os.makedirs(SRC_DIR, exist_ok=True)
		clone_repos(SRC_DIR)
		return

	# The default install branch
	if args.internal_stage is None:
		if not args.yes:
			try:
				click.confirm(f"Install FEniCS into {FENICS_DIR}?", abort=True)
				print("")
			except click.exceptions.Abort:
				return

		os.makedirs(FENICS_DIR, exist_ok=True)
		os.makedirs(SRC_DIR, exist_ok=True)

		# Don't clone if it was specified to use local repos
		if not args.local:
			clone_repos(SRC_DIR)

		# Create a virtual environment
		import virtualenv
		virtualenv.create_environment(VENV_DIR)
		print("Switching to virtual environment...")
		print("")

		print_stdout([os.path.join(VENV_DIR, "bin", "python3"), os.path.basename(sys.argv[0]), "--internal-stage", "1", *sys.argv[1:]], cwd=os.getcwd())
		print("")
		print_stdout([os.path.join(VENV_DIR, "bin", "python3"), os.path.basename(sys.argv[0]), "--internal-stage", "2", *sys.argv[1:]], cwd=os.getcwd())
		print("")
		print_stdout([os.path.join(VENV_DIR, "bin", "python3"), os.path.basename(sys.argv[0]), "--internal-stage", "3", *sys.argv[1:]], cwd=os.getcwd())
		print("")
		print("Done.")
		return

	# The second stage of the installation: running from virtualenv, install packages
	elif args.internal_stage == 1:
		pip_install(SRC_DIR)
		return

	# The third stage of the installation: build DOLFIN
	elif args.internal_stage == 2:
		pybind_build(SRC_DIR, BUILD_DIR, PYBIND_DIR)
		print("")
		dolfin_build(SRC_DIR, BUILD_DIR, VENV_DIR, DOLFIN_DIR, PYBIND_DIR, args.jobs)

	# The fourth stage: print information for the user
	elif args.internal_stage == 3:
		print("Installed packages in virtual environment:")
		pip.main(["list", "--format=columns"])

		print("")
		print(f"Activate FEniCS python virtualenv using 'source {FENICS_DIR}/fenics_env/bin/activate'")
		print(f"Activate DOLFIN build environment using 'source {FENICS_DIR}/dolfin/share/dolfin/dolfin.conf'")
		return
	
if __name__ == '__main__':
	main()