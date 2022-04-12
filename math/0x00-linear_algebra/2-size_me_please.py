#!/usr/bin/env python3

def matrix_shape(matrix):
    list1 = []
    
    try:
        x = len(matrix)
        list1.append(x)
    except:
        pass
    try:
        y = len(matrix[0])
        list1.append(y)
    except:
        pass
    try:
        z = len(matrix[0][0])
        list1.append(z)
    except:
        pass

    return list1