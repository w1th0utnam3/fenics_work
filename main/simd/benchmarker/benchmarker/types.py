import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

@dataclass
class FormTestData():
    form_name: str
    code_gen: Callable
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
class FormTestResult():
    avg: float
    min: float
    max: float
    speedup: float


@dataclass
class TestCase():
    compiler_args: List[Dict[str, Any]]
    run_args: List[TestRunArgs]
    forms: List[FormTestData]
    reference_case: Tuple[int, int]
    n_repeats: int


BenchmarkReport = Dict[str, Dict[Tuple[int, int], FormTestResult]]
