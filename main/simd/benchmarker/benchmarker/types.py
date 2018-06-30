import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# Types for test case definition

@dataclass
class FormTestData():
    form_name: str
    form_gen: Callable
    element_tensor_size: int
    coefficients: np.array
    coord_dofs: np.array
    n_elems: int


@dataclass
class TestRunArgs():
    name: str
    cross_element_width: int
    ffc_args: Dict[str, Any]


@dataclass
class TestCase():
    compiler_args: List[Dict[str, Any]]
    run_args: List[TestRunArgs]
    forms: List[FormTestData]
    reference_case: Tuple[int, int]
    n_repeats: int
    compiler_defines: Optional[List[Tuple[str, str]]] = None


# Types for benchmark results

@dataclass
class FormTestResult():
    avg: float
    min: float
    max: float
    speedup: float
    result_val: float
    result_ok: bool


@dataclass
class BenchmarkReport():
    total_runtime: float
    results: Dict[str, Dict[Tuple[int, int], FormTestResult]]
