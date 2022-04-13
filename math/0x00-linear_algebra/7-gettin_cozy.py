#!/usr/bin/env python3
"""Gettin’ Cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    result = []
    m1 = [x[:] for x in mat1]
    m2 = [x[:] for x in mat2]
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        result = m1 + m2
        return result
    else:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(m1)):
            result.append(m1[i] + m2[i])
        return result
