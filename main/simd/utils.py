import time
import string
import numpy as np

from typing import Iterable, Tuple

def replace_strings(text: str, replacement_pair_list: Iterable[Tuple[str, str]]) -> str:
    """Performs replacement of all tuples in the supplied list on the string"""

    new_text = text
    for old, new in replacement_pair_list:
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


def timing(n_runs: int, func, warm_up: bool = True, verbose: bool = True) -> Tuple[float, float, float]:
    """Measures avg, min and max execution time of 'func' over 'n_runs' executions"""

    lower = float('inf')
    upper = -float('inf')
    avg = 0

    if verbose:
        print(f"Timing (runs:{n_runs}): '{str(func)}' - ", end="", flush=True)

    # Call once without measurement "to get warm"
    if warm_up:
        if verbose:
            print("warm-up...", end="", flush=True)

        func()

        if verbose:
            print("done. ", end="", flush=True)

    for i in range(n_runs):
        start = time.time()
        func()
        end = time.time()

        diff = end - start

        lower = min(lower, diff)
        upper = max(upper, diff)
        avg += (diff - avg)/(i+1)

        if verbose:
            print("#", end="", flush=True)

    if verbose:
        print(" done.")

    return avg, lower, upper


def format_filename(s: str):
    """Take a string and return a valid filename constructed from the string"""
    # from: https://gist.github.com/seanh/93666

    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')  # I don't like spaces in filenames.
    return filename
