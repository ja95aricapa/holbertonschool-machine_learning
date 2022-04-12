#!/usr/bin/env python3
"""Flip Me Over"""

def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    # Zip returns an iterator of tuples, where the i-th tuple contains the i-th element 
    # from each of the argument sequences or iterables. 
    # In this example we unzip our array using * and then zip it to get the transpose.
    result = []
    if (isinstance(matrix, list) == True) and matrix:
            t_matrix = zip(*matrix)
            for row in t_matrix:
                result.append(row)

    return result
