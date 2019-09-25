from pal.constants.world import RANDOM_PERTERBATION_LIKELIHOOD

import scipy
import numpy as np


def solve_b_Ainv(b, A):
    '''
    Given a b and an A from the equation:

        Ax = b

    This returns what x is.  This is useful if we want to replace:

        x = bA^-1

    '''
    L = scipy.linalg.cho_factor(A, lower=True, overwrite_a=False)
    return scipy.linalg.cho_solve(L, b)


def gaussian(X, X_reduced, Y, mean, cov, hps, theta, force_rho_psd, indices, theta_flat):
    '''
    This function computes the likehood of solubilities given hyper parameters
    in the list theta.

    Note - when doing miso, X is a reduced list (say we sample 10 at each of 2 IS in the
    beginning, X is then 10 long, but Y is 20 long).

    **Parameters**

        X:
            A list of the sampled X coordinates.
        Y:
            A list of the objectives calculated, corresponding to the different X values.
        S:
            A list of lists, holding the solvent properties.
        mean:
            Function that, given X, Y, S, and theta, will calculate the mean vector.
        cov:
            Function that, given X, Y, S, and theta, will calculate the covariance matrix.
        hps:
            The one-d numpy array of keys for the hps.
        theta_flat:
            Object holding hyperparameters

    **Returns**

        likelihood: *float*
            The log of the likelihood without the constant term.
    '''

    # Wrap theta into the object
    theta.wrap({h: t for h, t in zip(hps, theta_flat)})

    if force_rho_psd:
        theta.gen_psd_rho()

    X = np.array([np.array(x) for x in X])
    X_reduced = np.array([np.array(x) for x in X_reduced])

    # Get the mean = [func() for y in Y]
    mu = mean(X, Y, theta)
    Sig = cov(X_reduced, Y, theta)
    if indices is not None:
        Sig = Sig[np.ix_(indices, indices)]

    y = Y - mu

    # Random perturbation for Sig so that we don't run into the issue of
    # having singular matrix during numerical optimization
    #rand_pert = np.random.random(Sig.shape) * RANDOM_PERTERBATION_LIKELIHOOD
    #L = scipy.linalg.cho_factor(Sig + rand_pert, lower=True, overwrite_a=False)

    L = scipy.linalg.cho_factor(Sig, lower=True, overwrite_a=False)
    alpha = scipy.linalg.cho_solve(L, y)

    val = -0.5 * y.T.dot(alpha) - sum([np.log(x) for x in np.diag(L[0])]) - len(mu) / 2.0 * np.log(2.0 * np.pi)

    return val


def bonilla(X, X_reduced, Y, mean, cov, hps, theta, force_rho_psd, indices, theta_flat):
    '''
    This function computes the likehood of solubilities given hyper parameters
    in the list theta.

    **Parameters**

        X:
            A list of the sampled X coordinates.
        Y:
            A list of the objectives calculated, corresponding to the different X values.
        S:
            A list of lists, holding the solvent properties.
        mean:
            Function that, given X, Y, S, and theta, will calculate the mean vector.
        cov:
            Function that, given X, Y, S, and theta, will calculate the covariance matrix.
        hps:
            The one-d numpy array of keys for the hps.
        theta_flat:
            Object holding hyperparameters

    **Returns**

        likelihood: *float*
            The log of the likelihood without the constant term.
    '''

    # Wrap theta into the object
    theta.wrap({h: t for h, t in zip(hps, theta_flat)})

    if force_rho_psd:
        theta.gen_psd_rho()

    X = np.array([np.array(x) for x in X])
    X_reduced = np.array([np.array(x) for x in X_reduced])

    mu = mean(X, Y, theta)
    Ks, Kx = cov(X_reduced, Y, theta, split=True)

    F = mean(X_reduced, Y, theta)
    F = np.repeat(F[:, np.newaxis], len(Ks), axis=1)

    FtKx_inv = solve_b_Ainv(F, Kx)
    FKs_inv = solve_b_Ainv(F.T, Ks)

    y = Y - mu
    D_inv = np.eye(len(y)) * 1E6
    M, N = len(Ks), len(Kx)

    val = -0.5 * N * np.log(np.linalg.norm(Ks))
    val += -0.5 * M * np.log(np.linalg.norm(Kx))
    val += -0.5 * np.trace(np.matmul(FtKx_inv, FKs_inv))
    val += -0.5 * np.trace(np.outer(y.T, np.outer(D_inv, y)))
    val += -0.5 * M * N * np.log(2.0 * np.pi)

    return val

def _test_gaussian():
    '''
    Unit test for the gaussian function.

    Checks for the following:

        1. Gaussian likelihood returns known value for known input.
    '''
    from pal.stats.hyperparameters import Theta
    from pal.kernels.matern import maternKernel52 as mk52

    ACTUAL = -6.9514227
    EPS = 1E-4

    hps = ['sig_m', 'l1', 'sig_beta', 'l2', 'mu_zeta', 'mu_alpha', 'sig_zeta', 'sig_alpha']
    theta_flat = [2.3, 1.0, 7.5, 1.0, 1E-3, 1.4, 27.2, 30.5]
    X = [
        [0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1.045,  40.24, 3.],
        [ 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.8075, 10.9, 5.]
    ]
    Y = [6.4537725, 9.94913233]
    theta = Theta()
    theta.wrap({h: t for h, t in zip(hps, theta_flat)})

    mean = lambda _1, Y, theta: np.array([4.0 * theta.mu_alpha + theta.mu_zeta for _ in Y])

    def cov(X, Y, theta):
        A = theta.sig_alpha * np.dot(np.array(X)[:, 1:-3], np.array(X)[:, 1:-3].T)
        B = theta.sig_beta * np.diag(np.ones(len(X)))
        C = theta.sig_zeta
        D = mk52(np.array(X)[:, -3:-1], [theta.l1, theta.l2], theta.sig_m)

        return A + B + C + D

    val = gaussian(X, Y, mean, cov, hps, theta, theta_flat)

    return abs(val - ACTUAL) < EPS


def run_unit_tests():
    assert _test_gaussian(), "pal.stats.likelihood.gaussian() has failed."

