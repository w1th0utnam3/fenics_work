from collections import namedtuple
from typing import Dict, Tuple


FormTestData = namedtuple("FormTestData", ["form_name",
                                           "code_gen",
                                           "element_tensor_size",
                                           "coefficients",
                                           "coord_dofs",
                                           "n_elems"])


TestRunArgs = namedtuple("TestRunArgs", ["name",
                                         "cross_element_width",
                                         "ffc_args"])


FormTestResult = namedtuple("FormTestResult", ["avg",
                                               "min",
                                               "max",
                                               "speedup"])


TestCase = namedtuple("TestCase", ["compiler_args",
                                   "run_args",
                                   "forms",
                                   "reference_case",
                                   "n_repeats"])


BenchmarkReport = Dict[str, Dict[Tuple[int, int], FormTestResult]]
