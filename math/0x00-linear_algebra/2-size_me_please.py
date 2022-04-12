#!/usr/bin/env python3

def matrix_shape(matrix):
    list1 = []
    if matrix:
        while (isinstance(matrix, list) == True):
            # check len of a dimension
            list1.append(len(matrix))
            # change to a next dimension on the loop
            matrix = matrix[0]

    return list1