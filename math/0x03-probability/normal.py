#!/usr/bin/env python3
"""Create a class Poisson that represents a poisson distribution"""


# constants
pi = 3.1415926536
e = 2.7182818285


# calculate fatorials
def factorial(n):
    """Finds the factorial of a given number"""
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)


class Poisson:
    """Represents a poisson distribution"""

    # task 0
    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson"""

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
            self.lambtha = sum(data) / len(data)

    # task 1
    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of successes (k)"""

        # If k is out of range
        if k < 0:
            return 0
        # f k is not an integer
        if type(k) != int:
            k = int(k)
        # calculate
        result = (e**(-self.lambtha) * self.lambtha**k) / factorial(k)
        return result

    # task 2
    def cdf(self, k):
        """Calculates the value of the CDF
        for a given number of successes (k)"""

        # if k is out of range
        if k < 0:
            return 0
        # if k is not an integer
        if type(k) != int:
            k = int(k)
        # calculate
        p = 0
        for i in range(k + 1):
            p += ((self.lambtha**i) / factorial(i))
        result = e**(-self.lambtha) * p
        return result
