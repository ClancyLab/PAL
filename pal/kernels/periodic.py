'''
The Periodic Kernel

This file contains functions for calculating the Periodic Kernel in order to get
a covariance function.  Note, the periodic kernel is geared towards modelling
periodic functions.  As such, it wouldn't really make sense in any other
situation.


**Functions**

    - perodicKernel(data, weights, sig):

'''

from numpy import exp, sin, sqrt, array, pi
from scipy.spatial import distance_matrix


def perodicKernel(data, weights, sig, p, other=None):
    '''
    Compute the periodic kernel from a data set, some weights for the different
    data points, and a variance.

    **Parameters**

        data: *list, list, float*
            A list of lists, holding floats which are our data points.

        weights: *list, float*
            A list of weights for the data points.

        sig: *float*
            A variance for our points.

        p: *float*
            The period of repetition.

        other: *list, list, float, optional*
            A list of lists, holding floats which are our data points. Note,
            this is used when wanting to compare against another range.

    **Returns**

        kernel: *list, list, float*
            An n-by-n matrix holding the kernel.  n is the number of data
            points in the input data object.
    '''
    # First step, let's ensure we have correct object
    if isinstance(weights, float):
        weights = [weights]
    data, weights = array(data), array(weights)**0.5
    n, dim = data.shape

    # Get the pairwise distance matrix between data points.  Note, because we
    # want to weight the distances based on each dimension, we first multiply
    # each dimension by the sqrt(weight)
    if other is None:
        pairwise_matrix = distance_matrix(data * weights, data * weights)
    else:
        pairwise_matrix = distance_matrix(data * weights, other * weights)

    # Now, we can get the kernel... not sure why sqrt(5)
    return sig * exp(-2.0 * sin(pairwise_matrix * pi / p)**2 / weights**2)


def run_unit_tests():
    pass
