import time
import numpy as np


def replace_strings(text: str, replacement_pair_list):
    """Performs replacement of all tuples in the supplied list on the string"""

    new_text = text
    for pair in replacement_pair_list:
        old, new = pair
        new_text = new_text.replace(old, new)
    return new_text


def scipy2numpy(scipy_obj):
    """Converts a scipy matrix to a numpy matrix"""

    scipy_mat = scipy_obj.mat()
    size = scipy_mat.getSize()
    np_mat = np.zeros(size)

    for i in range(size[0]):
        for j in range(size[1]):
            np_mat[i,j] = scipy_mat.getValue(i,j)

    return np_mat


def timing(n_runs: int, func, warm_up: bool = True):
    """Measures avg, min and max execution time of 'func' over 'n_runs' executions"""

    lower = float('inf')
    upper = -float('inf')
    avg = 0

    # Call once without measurement "to get warm"
    if warm_up:
        func()

    for i in range(n_runs):
        start = time.time()
        func()
        end = time.time()

        diff = end - start

        lower = min(lower, diff)
        upper = max(upper, diff)
        avg += (diff - avg)/(i+1)

    return avg, lower, upper
