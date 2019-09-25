from pal.stats.likelihood import gaussian as gaussian_loglike

import copy
import numpy as np
from scipy import optimize as op
import scipy.stats

from squid import doe_lhs


def _isFloat(v):
    try:
        float(v)
    except (ValueError, TypeError):
        return False
    return True


def MAP(X, Y, mean, cov, theta,
        prior=None, n_start=5, parallel=False, PROCESSES_ALLOWED=4,
        use_theta=False, force_rho_psd=False, loglike=gaussian_loglike):
    '''
    Maximum A Posteriori to determine what hyperparameters
    theta are good for a given X, Y input.

    **Parameters**

        data: *list, list, int/float*
            A set of data points of which we calculate the MAP from.

    **Returns**

        None
    '''

    TEST_NEW_METHOD = theta.n_IS > 1

    assert not parallel, "Error - We have not implemented parallel MAP yet!"

    # Get a list of everything
    theta.set_hp_names()
    hyperparams = theta.hp_names

    # Here we assign our bounds for the parameters needing optimization
    hold_bounds = tuple([theta.bounds[name] for name in hyperparams])
    bounds = [
        [bv if _isFloat(bv) else bv(X, Y) for bv in b]
        for b in hold_bounds
    ]

    # Select different starting hyperparameter sets.  Note, we select n_start
    # possible sets, and will take the best of them all
    if use_theta:
        init_values = [theta.unwrap()]
        n_start = 1
    else:
        sampled_values = doe_lhs.lhs(len(bounds), samples=n_start)
        init_values = [
            [s * (b[1] - b[0]) + b[0] for s, b in zip(sampled_values[j], bounds)]
            for j in range(n_start)
        ]

    # Initialize arrays
    map_list = np.zeros([n_start, len(bounds)])
    lkh_list = np.zeros(n_start)

    X0 = X
    indices=None

    if TEST_NEW_METHOD:
        # Get a list of all unique X, removing initial IS identifier
        X0 = []
        for x in X:
            if not any([all([a == b for a, b in zip(x[1:], xchk)]) for xchk in X0]):
                X0.append(x[1:])
        # Now, we get the sub-covariance matrix for the specified sampled X and Y
        indices = []
        for l in range(theta.n_IS):
            for i, x in enumerate(X0):
                test = [l] + list(x)
                if any([all([a == b for a, b in zip(test, xchk)]) for xchk in X]):
                    indices.append(l * len(X0) + i)

    # If no prior defined, use default from paper with a uniform prior for
    # the rho values if defined
    if prior is None:
        def prior(X, X0, Y, hyperparams, theta, theta_flat):
            theta.wrap({h: t for h, t in zip(hyperparams, theta_flat)})
            # From the paper, mean is the length of the bounds given
            mean = [sorted([bv if _isFloat(bv) else bv(X, Y)
                     for bv in theta.bounds[hp]])
                    for hp in hyperparams if "rho" not in hp]
            mean = np.array([upper - lower for lower, upper in mean])
            # Also from the misoKG paper
            std = np.array([m / 2.0 for m in mean])

            # For rho, we assume a flat prior
            # So... do nothing then?

            p = float(np.prod([scipy.stats.norm.pdf(x, loc=m, scale=s) for x, m, s in zip(theta_flat, mean, std)]))
            return p

    # MAP = Maximum A Posteriori.  But we use a minimizer! So invert the
    # likelihood instead.
    f = lambda *args: -1.0 * loglike(X, X0, Y, mean, cov, hyperparams, copy.deepcopy(theta), force_rho_psd, indices, *args) * prior(X, X0, Y, hyperparams, copy.deepcopy(theta), *args)

    for i in range(n_start):
        results = op.minimize(f, init_values[i], bounds=bounds)
        map_list[i, :] = results['x']  # Store the optimized parameters
        lkh_list[i] = results.fun  # Store the resulting likelihood

    # Now, select parameters for the max likelihood of these
    index = np.nanargmin(lkh_list)  # Note, min because we inverted the likelihood so we can use a minimizer.
    best_theta = map_list[index, :]

    # Read the last rhoHPs from best_theta into the rho dictionary
    theta.wrap({k: v for k, v in zip(hyperparams, best_theta)})

    if force_rho_psd:
        theta.gen_psd_rho()

    return theta


def run_unit_tests():
    pass

