'''
This module holds methods to calculate the Knowledge Gradient (KG).
'''
import scipy
import pandas
import warnings
import numpy as np
import scipy.stats


def compute_c_A(a_in, b_in):
    '''
    A generalized implementation of Algorithm 1 in Frazier 2009 paper for knowledge gradients.
    This code will return a sequence (c) and Acceptance set (A) from any two vectors a and
    b.  In regards to KG, a is the mean after n iterations, and b is the scaled diagonal of
    the covariance matrix by some x index (x in [0, n]).

    **Parameters**

         a_in: *list, float*
             The predicted mean at the nth iteration.

         b_in: *list, float*
             Scaled diagonal of the covariance matrix by some x index (x in [0, n])

    **Returns**

         c: *list, float*

         A: *list, int*

    '''
    M = len(a_in)
    # Use the same subscripts as in Algorithm 1, therefore, a[0] and b[0] are dummy values with no meaning
    a = np.concatenate(([np.inf], a_in))
    b = np.concatenate(([np.inf], b_in))
    c = np.zeros(M + 1)
    c[0] = -np.inf
    c[1] = np.inf
    A = [1]
    for i in range(1, M):
        c[i + 1] = np.inf
        while True:
            j = A[-1]
            c[j] = (a[j] - a[i + 1]) / (b[i + 1] - b[j])
            if len(A) != 1 and c[j] <= c[A[-2]]:
                del A[-1]
            else:
                break
        A.append(i + 1)
    return c, A


def compute_kg_new(a, b, cutoff=10.0):
    '''
    Algorithm 2 in Frazier 2009 paper.  This code itself is adapted from
    https://github.com/misokg/NIPS2017/blob/master/multifidelity_KG/voi/knowledge_gradient.py

    **Parameters**

        a:

        b:

        cutoff: *float, optional*

    **Returns**

        kg:
    '''

    assert len(a) == len(b), "Error - a and b should be the same length in compute_kg_new"

    if np.all(np.abs(b) <= 1e-6):
        return 0.0, np.array([]), 0.0

    df = pandas.DataFrame({'a': a, 'b': b})
    sorted_df = df.sort_values(by=['b', 'a'])
    sorted_df['drop_idx'] = np.zeros(len(sorted_df))
    sorted_index = sorted_df.index

    for i in xrange(len(sorted_index) - 1):
        if sorted_df.ix[sorted_index[i], 'b'] == sorted_df.ix[sorted_index[i + 1], 'b']:
            sorted_df.ix[sorted_index[i], 'drop_idx'] = 1

    truncated_df = sorted_df.ix[sorted_df['drop_idx'] == 0, ['a', 'b']]
    new_a = truncated_df['a'].values
    new_b = truncated_df['b'].values
    index_keep = truncated_df.index.values
    c, A = compute_c_A(new_a, new_b)

    if len(A) <= 1:
        return 0.0, np.array([]), 0.0

    final_b = np.array([new_b[idx - 1] for idx in A])
    final_index_keep = np.array([index_keep[idx - 1] for idx in A])
    final_c = np.array([c[idx] for idx in A])

    # compute log h() using numerically stable method
    d = np.log(final_b[1:] - final_b[:-1]) - 0.5 * np.log(2. * np.pi) - 0.5 * np.power(final_c[:-1], 2.0)
    abs_final_c = np.absolute(final_c[:-1])

    for i in xrange(len(d)):
        # Try to best approximate the value.  Note, if stat_val divides by 0, we get a runtime
        # warning.  As this is handled later on, we simply hide it here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat_val = scipy.stats.norm.cdf(-abs_final_c[i]) / scipy.stats.norm.pdf(abs_final_c[i])
        # If successful, and it is not inf, use it.
        if not np.isinf(stat_val) and not np.isnan(stat_val):
            val = np.log1p(-abs_final_c[i] * stat_val)
        else:
            # Otherwise, try another approximation.
            val = np.float64(final_c[i]) / np.float64(final_c[i] * final_c[i] + 1)
            val *= np.float64(final_c[i])
            # If the value is STILL absurd, then truncate/round to the best we can.
            val = min([0.9999999999999999, val])
            # Error handling for situations in which final_c[i] is insanely large suth that
            # val becomes essentially 1.0.
            val = np.log1p(np.float64(-val), dtype=np.float64)

        # If val is NaN, set to -inf
        if np.isnan(val):
            print("VAL IS NAN!")
            print "stat_val = %s" % str(stat_val)
            print "final_c[%d] = %s" % (i, str(final_c[i]))
            d[i] += float('-inf')
        else:
            d[i] += val

    kg = np.exp(d).sum()

    # If kg is calculated as nan, return -inf instead.
    if np.isnan(kg):
        kg = float('-inf')

    return kg, final_index_keep, abs_final_c


def _test_compute_kg_new():
    '''
    This runs a unit test on the KG factor calculation given an a and b vector.

    
    '''
    a = np.array([6.74, 6.56, 7.93, 6.51, 4.29])
    b = np.array([1.88, 1.30, 0.55, 0.21, 0.00]) / 1.88
    kg, final_index_keep, abs_final_c = compute_kg_new(a, b)

    expected = 0.0135125804399
    EPS = 1E-4

    return abs(expected - kg) < EPS

def run_unit_tests():
    assert _test_compute_kg_new(), "pal.stats.knowledge_gradient.compute_kg_new() failed."


if __name__ == "__main__":
    run_unit_tests()

