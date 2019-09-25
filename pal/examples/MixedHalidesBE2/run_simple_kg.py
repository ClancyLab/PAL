from pal.opt import Optimizer
from pal.constants.solvents import solvents
from pal.acquisition.kg import getNextSample_kg
from pal.kernels.matern import maternKernel52 as mk52
from pal.stats.MLE import MLE
from pal.stats.MAP import MAP

import os
import copy
import time
import numpy as np
import cPickle as pickle


def run_kg(run_index):
    # Store data for debugging
    IS0 = pickle.load(open("enthalpy_N1_R3_Ukcal-mol", 'r'))
    #IS0 = pickle.load(open("enthalpy_N3_R2_Ukcal-mol", 'r'))

    # Generate the main object
    sim = Optimizer()

    # Assign simulation properties
    #sim.hyperparameter_objective = MAP
    sim.hyperparameter_objective = MLE
    ###################################################################################################
    # File names
    sim.fname_out = "enthalpy_kg.dat"
    sim.fname_historical = "data_dumps/%d_reduced.history" % run_index

    print "Waiting on %s to be written..." % sim.fname_historical,
    while not os.path.exists(sim.fname_historical):
        time.sleep(30)
    print " DONE"

    # Information sources, in order from expensive to cheap
    sim.IS = [
        lambda h, c, s: -1.0 * IS0[' '.join([''.join(h), c, s])],
    ]
    sim.costs = [
        1.0
    ]
    sim.save_extra_files = True

    sim.logger_fname = "data_dumps/%d_kg.log" % run_index
    if os.path.exists(sim.logger_fname):
        os.system("rm %s" % sim.logger_fname)
    os.system("touch %s" % sim.logger_fname)

    sim.obj_vs_cost_fname = "data_dumps/%d_kg.dat" % run_index
    sim.mu_fname = "data_dumps/%d_mu_kg.dat" % run_index
    sim.sig_fname = "data_dumps/%d_sig_kg.dat" % run_index
    sim.sample_fname = "data_dumps/%d_sample_kg.dat" % run_index
    sim.combos_fname = "data_dumps/%d_combos_kg.dat" % run_index
    sim.hp_fname = "data_dumps/%d_hp_kg.dat" % run_index
    sim.acquisition_fname = "data_dumps/%d_acq_kg.dat" % run_index
    sim.historical_nsample = 10
    ########################################

    sim.n_start = 10  # The number of starting MLE samples
    sim.reopt = 20
    sim.ramp_opt = None
    sim.parallel = False

    sim.acquisition = getNextSample_kg

    # Possible compositions by default
    sim.A = ["Cs", "MA", "FA"]
    sim.B = ["Pb"]
    sim.X = ["Cl", "Br", "I"]
    sim.solvents = copy.deepcopy(solvents)
    sim.S = list(set([v["name"] for k, v in sim.solvents.items()]))
    sim.mixed_halides = True
    sim.mixed_solvents = False

    # Parameters for debugging and overwritting
    sim.debug = False
    sim.verbose = True
    sim.overwrite = True  # If True, warning, else Error

    # Functional forms of our mean and covariance
    # MEAN: 4 * mu_alpha + mu_zeta
    # COV: sig_alpha * |X><X| + sig_beta * I_N + sig_zeta + MaternKernel(S, weights, sig_m)

    SCALE = [2.0, 4.0][int(sim.mixed_halides)]
    # _1, _2, _3 used as dummy entries
    sim.mean = lambda _1, Y, theta: np.array([SCALE * theta.mu_alpha + theta.mu_zeta for _ in Y])

    def cov(X, Y, theta):
        A = theta.sig_alpha * np.dot(np.array(X)[:, 1:-3], np.array(X)[:, 1:-3].T)
        B = theta.sig_beta * np.diag(np.ones(len(X)))
        C = theta.sig_zeta
        D = mk52(np.array(X)[:, -3:-1], [theta.l1, theta.l2], theta.sig_m)

        return A + B + C + D

    sim.cov = cov

    sim.theta.bounds = {}
    sim.theta.mu_alpha, sim.theta.bounds['mu_alpha'] = None, (1E-3, lambda _, Y: max(Y))
    sim.theta.sig_alpha, sim.theta.bounds['sig_alpha'] = None, (1E-2, lambda _, Y: 10.0 * np.var(Y))
    sim.theta.sig_beta, sim.theta.bounds['sig_beta'] = None, (1E-2, lambda _, Y: 10.0 * np.var(Y))
    sim.theta.mu_zeta, sim.theta.bounds['mu_zeta'] = None, (1E-3, lambda _, Y: max(Y))
    sim.theta.sig_zeta, sim.theta.bounds['sig_zeta'] = None, (1E-2, lambda _, Y: 10.0 * np.var(Y))
    sim.theta.sig_m, sim.theta.bounds['sig_m'] = None, (1E-2, lambda _, Y: np.var(Y))
    sim.theta.l1, sim.theta.bounds['l1'] = None, (1E-1, 1)
    sim.theta.l2, sim.theta.bounds['l2'] = None, (1E-1, 1)

    # NOTE! This is a reserved keyword in misoKG.  We will generate a list of the same length
    # of the information sources, and use this for scaling our IS.
    # sim.theta.rho, sim.theta.bounds['rho'] = {"[0, 0]": 1}, (1E-1, 5.0)
    # NOTE! This is a reserved keyword in misoKG.  We will generate a list of the same length
    # of the information sources, and use this for scaling our IS.
    sim.theta.rho = {"[0, 0]": 1}
    sim.theta.bounds['rho [0, 0]'] = (1, 1)

    sim.theta.set_hp_names()

    sim.primary_rho_opt = False
    sim.update_hp_only_with_IS0 = False

    ###################################################################################################

    # Start simulation
    sim.run()

