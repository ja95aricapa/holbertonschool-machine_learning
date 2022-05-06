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
        """Calculates the cost of the model
        using logistic regression"""
        m = (Y.shape[1])
        return -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        # now, execute the predictions between n layers
        # using all the previous methods:
        #
        # X is a mtx that contains the input data
        # Y is a mtx that contains the correct labels for the input data
        #
        # so... with all the necessary data, we can begin:
        # 1) calculate the prediction (A) from the left layer
        # to the right layer
        # the output is a mtx containing the predicted
        # labels for each example
        # The label values should be 1 if the output
        # of the network is >= 0.5 and 0 otherwise
        pre = self.forward_prop(X)
        # 2) calculate the cost of the network
        # about do that prediction (activity)
        result_cost = self.cost(Y, pre)
        # 3) assure the format expected of the output
        pre_round = np.rint(pre)
        result_pre = pre_round.astype(int)
        return result_pre, result_cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        # Gradient descent is an iterative optimization algorithm,
        # which finds the minimum of a differentiable function.
        # This way, we can find an optimal solution minimizing
        # the cost over model parameters
        #
        # we would like to find how the cost changes with respect
        # to w and b, so as to change the original w and b slowly
        # to get the optimal parameters.
        #
        # m denotes the total number of training examples
        m = (Y.shape[1])
        # the way to fond the gradients is is to subtract from
        # the prediction its cost
        # but...
        # because A and Y are mtx with a lot of predictions
        # and costs, the way to operate all the difference
        # in one step is to the mtx A subtract the mtx Y
        dz = A - Y
        # After finding the gradients (dz), we need to subtract the
        # gradients with the original w and b. We subtract so
        # that we move the values of gradients in the opposite
        # direction to the slope so as to make sure the cost
        # is decreasing.
        # a way to find the adjusted values of weight and bias
        # is find the most efficient values of w and b that fits
        # with the prediction. arrange the respective values on mtx
        # for easy operation
        db = (1 / m) * np.sum(dz)
        dw = (1 / m) * np.matmul(X, dz.T)
        # Updates the private attributes
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db
