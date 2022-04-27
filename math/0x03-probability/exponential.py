#!/usr/bin/env python3
"""Create a class Exponential that represents an exponential distribution"""


# constants
pi = 3.1415926536
e = 2.7182818285


# calculate fatorials
def factorial(n):
    """Finds the factorial of a given number"""
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)


class Exponential:
    """Represents an exponential distribution"""

    # task 3
    def __init__(self, data=None, lambtha=1.):
        """Initialize Exponential"""

        # if data is not given
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            # Sets the instance attribute lambtha
            self.lambtha = float(lambtha)
        # if data is given
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            # Sets the instance attribute lambtha
            self.lambtha = 1 / (sum(data) / len(data))

    # task 4
    def pdf(self, x):
        """Calculates the value of the
        PDF for a given time period"""
        if x < 0:
            return 0

        return self.lambtha * e**(-self.lambtha * x)

    # task 5
    def cdf(self, x):
        """Calculates the value of the
        CDF for a given time period"""
        if x < 0:
            return 0

        return 1 - e**(-self.lambtha * x)
