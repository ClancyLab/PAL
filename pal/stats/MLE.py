from pal.stats.likelihood import gaussian as gaussian_loglike

import copy
import numpy as np
from scipy import optimize as op

from squid import doe_lhs


def _isFloat(v):
    try:
        float(v)
    except (ValueError, TypeError):
        return False
    return True


def MLE(X, Y, mean, cov, theta,
        prior=None, n_start=5, parallel=False, PROCESSES_ALLOWED=4,
        use_theta=False, force_rho_psd=False, loglike=gaussian_loglike):
    '''
    Maximum Likelihood Estimation to determine what hyperparameters
    theta are good for a given X, Y input.

    **Parameters**

        data: *list, list, int/float*
            A set of data points of which we calculate the MLE from.

    **Returns**

        None
    '''

    get_reduced_X = theta.n_IS > 1

    assert not parallel, "Error - We have not implemented parallel MLE yet!"

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
    sampled_values = doe_lhs.lhs(len(bounds), samples=n_start)
    init_values = [
        [s * (b[1] - b[0]) + b[0] for s, b in zip(sampled_values[j], bounds)]
        for j in range(n_start)
    ]
    if use_theta:
        init_values.append(theta.unwrap())
        n_start += 1

    # Initialize arrays
    mle_list = np.zeros([n_start, len(bounds)])
    lkh_list = np.zeros(n_start)

    X0 = X
    indices=None

    if get_reduced_X:
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

    # MLE = Maximum Likelihood Estimation.  But we use a minimizer! So invert the
    # likelihood instead.
    f = lambda *args: -1.0 * loglike(X, X0, Y, mean, cov, hyperparams, copy.deepcopy(theta), force_rho_psd, indices, *args)

    for i in range(n_start):
        results = op.minimize(f, init_values[i], bounds=bounds)
        mle_list[i, :] = results['x']  # Store the optimized parameters
        lkh_list[i] = results.fun  # Store the resulting likelihood

    # Now, select parameters for the max likelihood of these
    index = np.nanargmin(lkh_list)  # Note, min because we inverted the likelihood so we can use a minimizer.
    best_theta = mle_list[index, :]

    # Read the last rhoHPs from best_theta into the rho dictionary
    theta.wrap({k: v for k, v in zip(hyperparams, best_theta)})

    if force_rho_psd:
        theta.gen_psd_rho()

    return theta


def run_unit_tests():
    pass
