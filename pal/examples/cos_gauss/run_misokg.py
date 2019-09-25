from pal.opt import Optimizer
import pal.utils.strings as pal_strings
from pal.constants.solvents import solvents
from pal.kernels.matern import maternKernel52 as mk52
from pal.kernels.squared import squared
from pal.acquisition.misokg import getNextSample_misokg
from pal.stats.MLE import MLE
from pal.stats.MAP import MAP

import os
import copy
# import random
import numpy as np
import cPickle as pickle


def run_misokg(run_index, sffx="misokg", scaled=False, loose=False,
               very_loose=False, use_MAP=False, upper=1.0,
               unitary=None, use_miso=False):

    SAMPLE_DOMAIN = 1000

    # Generate the main object
    sim = Optimizer()

    # Assign simulation properties
    if use_MAP:
        sim.hyperparameter_objective = MAP
    else:
        sim.hyperparameter_objective = MLE
    ###################################################################################################
    # File names
    sim.fname_out = None
    sim.fname_historical = None

    sim.logger_fname = "data_dumps/%d_%s.log" % (run_index, sffx)
    if os.path.exists(sim.logger_fname):
        os.system("rm %s" % sim.logger_fname)
    os.system("touch %s" % sim.logger_fname)

    sim.obj_vs_cost_fname = None
    sim.mu_fname = None
    sim.sig_fname = None
    sim.combos_fname = None
    #sim.hp_fname = None
    sim.hp_fname = "data_dumps/%d_HP_%s.log" % (run_index, sffx)
    sim.acquisition_fname = None
    sim.save_extra_files = True

    # Information sources, in order from expensive to cheap
    gauss = lambda x1, x2: -1.0 * np.exp(-0.5 * (x1**2 + x2**2))
    cos = lambda x1, x2: -1.0 * np.cos(x1 * x2)
    sim.IS = [
        gauss,
        cos
    ]
    sim.costs = [
        1000.0,
        1.0
    ]

    ########################################
    sim.numerical = True
    sim.historical_nsample = 5
    sim.domain = [
        (-2.0, 2.0),
        (-2.0, 2.0)
    ]
    sim.sample_n_from_domain = SAMPLE_DOMAIN
    ########################################
    
    sim.n_start = 10  # The number of starting MLE samples
    sim.reopt = 20
    sim.ramp_opt = None
    sim.parallel = False

    # Parameters for debugging and overwritting
    sim.debug = False
    sim.verbose = True
    sim.overwrite = True  # If True, warning, else Error

    sim.acquisition = getNextSample_misokg

    # Functional forms of our mean and covariance
    sim.mean = lambda X, Y, theta: np.array([-1.0 for _ in Y])

    def cov_miso(X0, Y, theta):
        Kx = squared(np.array(X0)[:, 1:], [theta.l1, theta.l2], theta.sig_1)
        Kx_l = squared(np.array(X0)[:, 1:], [theta.l3, theta.l4], theta.sig_2)
        return np.block([[Kx, Kx], [Kx, Kx + Kx_l]])

    def cov_bonilla(X0, Y, theta):
        #Kx = mk52(np.array(X0)[:, 1:], [theta.l1, theta.l2], theta.sig_m)
        Kx = squared(np.array(X0)[:, 1:], [theta.l1, theta.l2], theta.sig_m)
        Kx = Kx + 1E-6 * np.eye(Kx.shape[0])

        if unitary is not None:
            Ks = np.array([[1.0, float(unitary)], [float(unitary), 1.0]])
        else:
            L = np.array([
                np.array([theta.rho[str(sorted([i, j]))] if i >= j else 0.0 for j in range(theta.n_IS)])  # Lower triangulary
                for i in range(theta.n_IS)
            ])
            Ks = L.dot(L.T)

        e = np.diag(np.array([theta.e1, theta.e2]))

        Ks = np.matmul(e, np.matmul(Ks, e))
        
        K = np.kron(Ks, Kx)

        return np.kron(Ks, Kx)

    if use_miso:
        sim.cov = cov_miso

        sim.theta.bounds = {}
        sim.theta.sig_1, sim.theta.bounds['sig_1'] = None, (1E-2, lambda _, Y: np.var(Y))
        sim.theta.sig_2, sim.theta.bounds['sig_2'] = None, (1E-2, lambda _, Y: np.var(Y))
        sim.theta.l1, sim.theta.bounds['l1'] = None, (1E-1, 1)
        sim.theta.l2, sim.theta.bounds['l2'] = None, (1E-1, 1)
        sim.theta.l3, sim.theta.bounds['l3'] = None, (1E-1, 1)
        sim.theta.l4, sim.theta.bounds['l4'] = None, (1E-1, 1)

        sim.theta.rho = {"[0, 0]": 1.0, "[0, 1]": 1.0, "[1, 1]": 1.0}
        sim.theta.bounds['rho [0, 0]'] = (1.0, 1.0)
        sim.theta.bounds['rho [0, 1]'] = (1.0, 1.0)
        sim.theta.bounds['rho [1, 1]'] = (1.0, 1.0)
    else:
        sim.cov = cov_bonilla

        sim.theta.bounds = {}
        sim.theta.sig_m, sim.theta.bounds['sig_m'] = None, (1E-2, lambda _, Y: np.var(Y))
        sim.theta.l1, sim.theta.bounds['l1'] = None, (1E-1, 1)
        sim.theta.l2, sim.theta.bounds['l2'] = None, (1E-1, 1)
    
        if scaled:
            sim.theta.e1, sim.theta.bounds['e1'] = None, (1E-1, upper)
            sim.theta.e2, sim.theta.bounds['e2'] = None, (1E-1, upper)
        else:
            sim.theta.e1, sim.theta.bounds['e1'] = 1.0, (1E-1, upper)
            sim.theta.e2, sim.theta.bounds['e2'] = 1.0, (1E-1, upper)
    
        # NOTE! This is a reserved keyword in misoKG.  We will generate a list of the same length
        # of the information sources, and use this for scaling our IS.
        if very_loose:
            sim.theta.rho = {"[0, 0]": None, "[0, 1]": None, "[1, 1]": None}
        elif loose:
            sim.theta.rho = {"[0, 0]": 1.0, "[0, 1]": None, "[1, 1]": 1.0}
        elif unitary is not None:
            sim.theta.rho = {"[0, 0]": 1.0, "[0, 1]": float(unitary), "[1, 1]": 1.0}
        else:
            raise Exception("What is trying to be run?")
        sim.theta.bounds['rho [0, 0]'] = (0.1, upper)
        sim.theta.bounds['rho [0, 1]'] = (0.1, upper)
        sim.theta.bounds['rho [1, 1]'] = (0.01, upper)

    sim.theta.set_hp_names()

    sim.update_hp_only_with_IS0 = False
    sim.update_hp_only_with_overlapped = False
    sim.theta.normalize_L = False
    sim.theta.normalize_Ks = False

    ###################################################################################################

    # Start simulation
    sim.iteration_kill_switch = 200
    sim.cost_kill_switch = 10000
    sim.run()


if __name__ == "__main__":
    run_misokg(6660, sffx="debug", scaled=True, loose=False, very_loose=True,
               use_MAP=False, upper=1.0, unitary=None, use_miso=False)

