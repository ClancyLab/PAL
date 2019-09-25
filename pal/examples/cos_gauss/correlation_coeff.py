import numpy as np
from squid import doe_lhs
from scipy.stats import pearsonr

gauss = lambda x1, x2: -1.0 * np.exp(-0.5 * (x1**2 + x2**2))
cos = lambda x1, x2: -1.0 * np.cos(x1 * x2)

IS = [
    gauss,
    cos
]

domain = [
    (-2.0, 2.0),
    (-2.0, 2.0)
]

IS = [
    gauss,
    cos
]

NSAMPLES = 1000
samples = doe_lhs.lhs(len(domain), NSAMPLES)

Y = [
    [I(*x) for x in samples]
    for I in IS
]

print pearsonr(Y[0], Y[1])

