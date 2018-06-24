from collections import namedtuple

FormTestData = namedtuple("FormTestData", ["form_name",
                                           "code_gen",
                                           "element_tensor_size",
                                           "coefficients",
                                           "coord_dofs",
                                           "n_elems"])


TestRunArgs = namedtuple("TestRunArgs", ["name",
                                         "cross_element_width",
                                         "ffc_args"])


TestCase = namedtuple("TestCase", ["compiler_args",
                                   "run_args",
                                   "forms",
                                   "reference_case",
                                   "n_repeats"])


FormTestResult = namedtuple("FormTestResult", ["avg",
                                               "min",
                                               "max",
                                               "speedup"])