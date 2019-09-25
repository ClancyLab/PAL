'''
This file contains the Expected Improvement (EI) method, in which the next choice is
based on the maximum of the difference between predicted and best observed value (with
some other math accounted for)
'''
import numpy as np
from scipy.stats import norm


def getNextSample_EI(mu, K, best, dim, _1, _2, samples, save=None):
    '''
    This function determines the next sample to run based on the Expected Improvement
    method.
    '''
    EI_list = np.array([
        (mu[i] - best) * norm.cdf((mu[i] - best) / np.sqrt(K[i, i])) +
        np.sqrt(K[i, i]) * norm.pdf((mu[i] - best) / np.sqrt(K[i, i]))
        if (K[i, i] > 0 and i not in samples) else 0
        for i in range(dim)]).reshape((-1, dim))[0]
    next_sample = np.nanargmax(EI_list)

    if save is not None:
        fptr = open(save, 'a')
        fptr.write('\n' + " ".join(["%f" % ei for ei in EI_list]))
        fptr.close()

    if np.nanmax([EI_list[next_sample], 0]) <= 0:
        return np.random.choice([i for i in range(len(mu)) if i not in samples])

    return next_sample


def run_unit_tests():
    pass
