'''
The Matern Kernel

This file contains functions for calculating the Matern Kernel in order to get
a covariance function.  Note, the matern kernel is based on spatial distances
between points, and as such would only really make sense in trying to capture
systems in which spatial differences matter.

Interesting to note, as the smoothing term v goes to 0.5 we get the exponential
kernel, and as it goes to inf we get the Gaussian Kernel.

**Functions**

    - maternKernel52(data, weights, sig):

'''

from numpy import exp, sqrt, array
from scipy.spatial import distance_matrix


def maternKernel52(data, weights, sig, other=None):
    '''
    Compute the 5/2 matern kernel from a data set, some weights for the different
    data points, and a variance.

    **Parameters**

        data: *list, list, float*
            A list of lists, holding floats which are our data points.

        weights: *list, float*
            A list of weights for the data points

        sig: *float*
            A variance for our points

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
    K = sqrt(5) * pairwise_matrix
    return sig * (1.0 + K + K**2 / 3.0) * exp(-K)


def run_unit_tests():
    pass
