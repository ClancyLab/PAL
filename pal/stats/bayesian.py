import numpy as np


def sequential_posterior_update(x, y, mu, K, err=1E-6):
    '''
    Do a sequential bayesian posterior update, using the Kalman Filter method.
    Our Kalman Gain is defined as:

        KG = K[x, :] / (err + K[x, x])

    And the update is then:

        mu_new = mu_old + KG * (y_observered - mu_predicted_at_x)
    '''

    # If the variance is 0 before we account for it, throw an error!
    assert K[x, x] > 0, "Error - Variance is 0!  Possibly double counted a point."

    cov_vec = K[x]
    mu_new = mu + (y - mu[x]) / (K[x, x] + err) * cov_vec
    Sig_new = K - np.outer(K[x, :], K[:, x]) / (K[x, x] + err)

    return mu_new, Sig_new


def full_posterior_update():
    '''
    Update based on the Bayesian Optimization in which only observed points
    are taken into account.
    '''
    raise Exception("This has not yet been implemented!")


def _test_sequential_posterior_update():
    '''
    Unit test for the sequential_posterior_update function.

    Checks for the following:

        1. Can re iteratively update all of the domain without crashing.
        2. Afterwards, are all variances non-negative and small
    '''
    from pal.stats.hyperparameters import Theta
    from pal.kernels.matern import maternKernel52 as mk52

    hps = ['sig_m', 'l1', 'sig_beta', 'l2', 'mu_zeta', 'mu_alpha', 'sig_zeta', 'sig_alpha']
    theta_flat = [2.3, 1.0, 7.5, 1.0, 1E-3, 1.4, 27.2, 30.5]
    X = [
        [0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1.045,  40.24, 3.],
        [ 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.8075, 10.9, 5.],
        [ 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0.8075, 10.9, 5.],
        [ 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.8075, 10.9, 5.]
    ]
    Y = [6.4537725, 9.94913233, 8.0, 12]
    theta = Theta()
    theta.wrap({h: t for h, t in zip(hps, theta_flat)})

    mean = lambda _1, Y, theta: np.array([4.0 * theta.mu_alpha + theta.mu_zeta for _ in Y])

    def cov(X, Y, theta):
        A = theta.sig_alpha * np.dot(np.array(X)[:, 1:-3], np.array(X)[:, 1:-3].T)
        B = theta.sig_beta * np.diag(np.ones(len(X)))
        C = theta.sig_zeta
        D = mk52(np.array(X)[:, -3:-1], [theta.l1, theta.l2], theta.sig_m)

        return A + B + C + D

    mu = mean(X, Y, theta)
    K = cov(X, Y, theta)

    # Check if we can update all posteriors without crashing
    for i, y in enumerate(Y):
        mu, K = sequential_posterior_update(i, y, mu, K)

    # Ensure diagonal is all possitive AND small (already all sampled)
    return all([k >= 0 for k in np.diag(K)]) and all([k < 1E-4 for k in np.diag(K)])


def run_unit_tests():
    assert _test_sequential_posterior_update(), "pal.stats.bayesian.sequential_posterior_update() has failed."
