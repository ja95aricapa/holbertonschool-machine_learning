#!/usr/bin/env python3

def matrix_shape(matrix):
    x = len(matrix)
    y = len(matrix[0])
    list1 = [x, y]
    try:
        z = len(matrix[0][0])
        list1.append(z)
    except:
        pass
    return list1