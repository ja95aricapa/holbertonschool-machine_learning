#!/usr/bin/env python3
"""Normal distribution"""


# constants
pi = 3.1415926536
e = 2.7182818285


# constants
def erf(x):
    """error function encountered in
    integrating the normal distribution"""
    a = (2 / (pi**(1/2)))
    b = (x - ((x**3)/3) + ((x**5)/10) - ((x**7)/42) + ((x**9)/216))
    return a * b


# task 6
class Normal:
    """Represents an normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal"""
        self.data = data

        # If data is not given
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
            self.mean = float(mean)

        # If data is given
        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            # calculate mean and stddev
            self.mean = sum(data) / len(data)

            s_dif = []
            for d in data:
                s_dif.append((d - self.mean)**2)

            self.stddev = (sum(s_dif) / len(s_dif))**(1/2)

    # task 7
    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        result = (x - self.mean) / self.stddev
        return result

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        result = self.stddev * z + self.mean
        return result

    # task 8
    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        aux = ((x - self.mean) / self.stddev)**2
        result = (1 / (self.stddev * (2 * pi)**(1/2))) * e**((-1/2) * aux)
        return result

    # task 9
    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        aux = (x - self.mean)/(self.stddev * (2**(1/2)))
        result = (1/2) * (1 + erf(aux))
        return result
