#!/usr/bin/env python3
"""Neuron"""
import numpy as np


# utils functions
def sigmoid(z):
    """apply sigmoid activation method"""
    result = 1 / (1 + np.exp(-z))
    return result


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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """weights vector for the neuron"""
        return self.__W

    @property
    def b(self):
        """bias for the neuron"""
        return self.__b

    @property
    def A(self):
        """activated output of the neuron (prediction)"""
        return self.__A

    def forward_prop(self, X):
        """calculates the forward propagation of the neuron"""
        # Forward Propagation is the way to move from the Input layer (left)
        # to the Output layer (right) in the neural network
        #
        # multiply the matrix input data (left layer) with the weights vector
        # (each node connection with the right) for the neuron
        #
        z = np.matmul(self.__W, X)
        # z is the activity of the next layer
        # one row for each example and one column for each hidden unit
        #
        # add the bias for the neuron to the activity
        z += self.__b
        # apply the activation function
        self.__A = sigmoid(z)
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model
        using logistic regression
        """
        # the cost function measures how well our
        # parameters w and b are doing on the training data set
        #
        # m denotes the total number of training examples
        m = (Y.shape[1])
        # for a log regrex, the loss function definition
        #
        # * Y is a mtx that contains the correct labels
        # for the input data (be one or the other class)
        # * A is a mtx containing the activated output
        # of the neuron for each example
        # * log(A) or Log(1-A) is the probability of be one o other label
        # ** in this case: A -> label:1 and (1-A) -> label:0
        #
        # for a single training example is:
        una = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        # now, for m number of training examples sum each result
        suma = np.sum(una)
        # finally, set the cost for every each example
        result = -(1 / m) * suma
        # the suggested formula for this task is:
        # -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        #
        return result
