from pal.constants.world import RANDOM_PERTERBATION_LIKELIHOOD

import scipy
import numpy as np


def uniform(hps, theta, theta_flat):
    '''
    This function ...

    **Parameters**

        hps:
            The one-d numpy array of keys for the hps.
        theta:
            The Theta hyperparameter object.
        theta_flat:
            Flat 1D array of hyperparameters.

    **Returns**

    '''

    # Wrap theta into the object
    theta.wrap({h: t for h, t in zip(hps, theta_flat)})

    raise Exception("HASN'T BEEN DONE YET!")


def run_unit_tests():
    pass
