import numpy as np
from squid import doe_lhs
from scipy.stats import pearsonr

rosenbrock = lambda x1, x2: (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3

IS = [
    lambda x1, x2: -1.0 * rosenbrock(x1, x2),
    lambda x1, x2: -1.0 * (rosenbrock(x1, x2) + 0.1 * np.sin(10.0 * x1 + 5.0 * x2))
]

domain = [
    (-2.0, 2.0),
    (-2.0, 2.0)
]

NSAMPLES = 1000
samples = doe_lhs.lhs(len(domain), NSAMPLES)

Y = [
    [I(*x) for x in samples]
    for I in IS
]

print pearsonr(Y[0], Y[1])


