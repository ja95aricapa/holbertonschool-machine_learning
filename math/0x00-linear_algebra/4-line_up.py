#!/usr/bin/env python3
"""Line Up"""

def add_arrays(arr1, arr2):
    """Adds two arrays element-wise"""
    result = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            result.append(arr1[i] + arr2[i])
        return result
    else:
        return None