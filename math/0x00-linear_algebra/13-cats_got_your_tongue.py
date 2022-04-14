#!/usr/bin/env python3
"""Cat's Got Your Tongue"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices"""
    new_mat = np.concatenate((mat1, mat2), axis)
    return new_mat
