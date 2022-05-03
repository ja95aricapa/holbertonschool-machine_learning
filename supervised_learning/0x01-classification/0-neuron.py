#!/usr/bin/env python3
"""Neuron"""
import numpy as np


# task 10
class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Class constructor"""

        # check if nx is a valid input features to the neuron
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        # set values
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
