import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Types for test case definition

# Data that defines an UFL form for benchmarking
@dataclass
class FormTestData():
    # Name of the form.
    form_name: str
    # Form code, tuple of Python code strings: (form expression, expression environment)
    #   The latter is evaluated using exec() and the generated globals dict passed as the
    #   globals argument when evaluating the expression with eval(). The value of the expression
    #   should be an UFL.Form and is used for benchmarking.
    form_code: Tuple[str, str]
    # Total number of values in the element tensor of the form.
    element_tensor_size: int
    # Numpy array that is used as coefficient input to the tabulate_tensor function of the form during benchmarking.
    coefficients: np.array
    # Coordinate array that is used as input to the tabulate_tensor function of the form during benchmarking.
    coord_dofs: np.array
    # Number of times the tabulate_tensor function should be called during benchmarking
    n_elems: int


# Set of parameters that are relevant during the code generation stage of benchmarking (e.g. for FFC)
@dataclass
class TestRunParameters():
    # Name of the parameter
    name: str
    # The cross element width for the tabulate_tensor function
    cross_element_width: int
    # Additional argument tuples for FFC
    ffc_args: Dict[str, Any]


# A test case that completely defines a benchmark session with n_forms x n_run_args x n_compiler_args test calls
@dataclass
class TestCase():
    # Parameter sets that are supplied to the C compiler
    compiler_args: List[Dict[str, Any]]
    # Parameter sets that are passed to FFC or used when calling a compiled benchmark test
    run_args: List[TestRunParameters]
    # Form definitions
    forms: List[FormTestData]
    # Index tuple into the compiler and run parameter lists for the combination which is used as reference for speedup
    reference_case: Tuple[int, int]
    # How many time a test should be repeated for the calculation of average runtime
    n_repeats: int
    # Optional macro defines that are passed to the compiler
    compiler_defines: Optional[List[Tuple[str, str]]] = None


# Types for benchmark results

# Result of a single form benchmark test
@dataclass
class FormTestResult():
    # Average runtime
    avg: float
    # Minimum runtime
    min: float
    # Maximum runtime
    max: float
    # Speedup relative to reference test
    speedup: float
    # The result of the reduction of kernel outputs
    result_val: float
    # Flag whether the result is close to the result of the reference test
    result_ok: bool


@dataclass
class BenchmarkReport():
    # The total runtime in seconds of the whole test case
    total_runtime: float
    # Dict of form names -> dict of index tuples (i,j),
    #   where i is compiler parameter set and j is run parameter set -> the specific result
    results: Dict[str, Dict[Tuple[int, int], FormTestResult]]
