import numpy as np


def replace_strings(text: str, replacement_pair_list):
    """Performs replacement of all tuples in the supplied list on the string"""

    new_text = text
    for pair in replacement_pair_list:
        old, new = pair
        new_text = new_text.replace(old, new)
    return new_text


def scipy2numpy(scipy_mat):
    """Converts a scipy matrix to a numpy matrix"""

    size = scipy_mat.getSize()
    np_mat = np.zeros(size)

    for i in range(size[0]):
        for j in range(size[1]):
            np_mat[i,j] = scipy_mat.getValue(i,j)

    return np_mat
