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


def run_ei(run_index, SAMPLE_DOMAIN=1000):

    FOLDER = "RNS%d" % SAMPLE_DOMAIN
    sffx = "ei"

    # Generate the main object
    sim = Optimizer()

    # Assign simulation properties
    #if use_MAP:
    #    sim.hyperparameter_objective = MAP
    #else:
    sim.hyperparameter_objective = MLE
    ###################################################################################################
    # File names
    sim.fname_out = None
    sim.fname_historical = None

    sim.logger_fname = "%s/%d_%s.log" % (FOLDER, run_index, sffx)
    if os.path.exists(sim.logger_fname):
        os.system("rm %s" % sim.logger_fname)
    os.system("touch %s" % sim.logger_fname)

    sim.obj_vs_cost_fname = None
    sim.mu_fname = None
    sim.sig_fname = None
    sim.combos_fname = None
    sim.hp_fname = None
    sim.acquisition_fname = None
    sim.save_extra_files = True

    # Information sources, in order from expensive to cheap
    rosenbrock = lambda x1, x2: (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3
    sim.IS = [
        lambda x1, x2: -1.0 * rosenbrock(x1, x2)
    ]
    sim.costs = [
        1000.0
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

    # Functional forms of our mean and covariance
    sim.mean = lambda X, Y, theta: np.array([-456.3 for _ in Y])

    def cov(X0, Y, theta):
        return squared(np.array(X0)[:, 1:], [theta.l1, theta.l2], theta.sig_1)

    sim.cov = cov

    sim.theta.bounds = {}
    sim.theta.sig_1, sim.theta.bounds['sig_1'] = None, (1E-2, lambda _, Y: np.var(Y))
    sim.theta.l1, sim.theta.bounds['l1'] = None, (1E-1, 1)
    sim.theta.l2, sim.theta.bounds['l2'] = None, (1E-1, 1)
    sim.theta.rho = {str(sorted([i, j])): 1.0 for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
    for k in sim.theta.rho.keys():
        sim.theta.bounds['rho %s' % k] = (0.1, 1.0)
        a, b = eval(k)
        if a != b:
            sim.theta.bounds['rho %s' % k] = (0.01, 1.0 - 1E-6)

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

