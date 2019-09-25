from pal.opt import Optimizer
import pal.utils.strings as pal_strings
from pal.constants.solvents import solvents
from pal.kernels.matern import maternKernel52 as mk52
# from pal.objectives.binding_energy import get_binding_energy as BE
from pal.acquisition.misokg import getNextSample_misokg
from pal.stats.MLE import MLE
from pal.stats.MAP import MAP

import os
import copy
# import random
import numpy as np
import cPickle as pickle


def run_misokg(run_index):

    # Store data for debugging
    IS0 = pickle.load(open("enthalpy_N1_R3_Ukcal-mol", 'r'))
    IS1 = pickle.load(open("enthalpy_N1_R2_Ukcal-mol", 'r'))

    # Generate the main object
    sim = Optimizer()

    # Assign simulation properties
    #sim.hyperparameter_objective = MAP
    sim.hyperparameter_objective = MLE
    ###################################################################################################
    # File names
    sim.fname_out = "enthalpy_misokg.dat"
    sim.fname_historical = None

    # Information sources, in order from expensive to cheap
    sim.IS = [
        lambda h, c, s: -1.0 * IS0[' '.join([''.join(h), c, s])],
        lambda h, c, s: -1.0 * IS1[' '.join([''.join(h), c, s])]
    ]
    sim.costs = [
        1.0,
        0.1
    ]

    sim.logger_fname = "data_dumps/%d_misokg.log" % run_index
    if os.path.exists(sim.logger_fname):
        os.system("rm %s" % sim.logger_fname)
    os.system("touch %s" % sim.logger_fname)

    sim.obj_vs_cost_fname = "data_dumps/%d_misokg.dat" % run_index
    sim.mu_fname = "data_dumps/%d_mu_misokg.dat" % run_index
    sim.sig_fname = "data_dumps/%d_sig_misokg.dat" % run_index
    sim.combos_fname = "data_dumps/%d_combos_misokg.dat" % run_index
    sim.hp_fname = "data_dumps/%d_hp_misokg.dat" % run_index
    sim.acquisition_fname = "data_dumps/%d_acq_misokg.dat" % run_index
    sim.save_extra_files = True
    ########################################
    # Override the possible combinations with the reduced list of IS0
    # Because we do this, we should also generate our own historical sample
    combos_no_IS = [k[1] + "Pb" + k[0] + "_" + k[2] for k in [key.split() for key in IS0.keys()]]
    sim.historical_nsample = 10
    choices = np.random.choice(combos_no_IS, sim.historical_nsample, replace=False)
    tmp_data = pal_strings.alphaToNum(
        choices,
        solvents,
        mixed_halides=True,
        name_has_IS=False)

    data = []
    for IS in range(len(sim.IS)):
        for i, d in enumerate(tmp_data):
            h, c, _, s, _ = pal_strings.parseName(pal_strings.parseNum(d, solvents, mixed_halides=True, num_has_IS=False), name_has_IS=False)
            c = c[0]
            data.append([IS] + d + [sim.IS[IS](h, c, s)])

    sim.fname_historical = "data_dumps/%d.history" % run_index
    pickle.dump(data, open(sim.fname_historical, 'w'))
    simple_data = [d for d in data if d[0] == 0]
    pickle.dump(simple_data, open("data_dumps/%d_reduced.history" % run_index, 'w'))

    ########################################

    sim.n_start = 10  # The number of starting MLE samples
    sim.reopt = 20
    sim.ramp_opt = None
    sim.parallel = False

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

    sim.acquisition = getNextSample_misokg

    # Functional forms of our mean and covariance
    # MEAN: 4 * mu_alpha + mu_zeta
    # COV: sig_alpha * |X><X| + sig_beta * I_N + sig_zeta + MaternKernel(S, weights, sig_m)

    SCALE = [2.0, 4.0][int(sim.mixed_halides)]
    # _1, _2, _3 used as dummy entries
    def mean(X, Y, theta):
        mu = np.array([SCALE * theta.mu_alpha + theta.mu_zeta for _ in Y])
        return mu
    sim.mean = mean

    def cov_old(X, Y, theta):
        A = theta.sig_alpha * np.dot(np.array(X)[:, 1:-3], np.array(X)[:, 1:-3].T)
        B = theta.sig_beta * np.diag(np.ones(len(X)))
        C = theta.sig_zeta
        D = mk52(np.array(X)[:, -3:-1], [theta.l1, theta.l2], theta.sig_m)
        return theta.rho_matrix(X) * (A + B + C + D)

    def cov(X0, Y, theta):
        A = theta.sig_alpha * np.dot(np.array(X0)[:, :-3], np.array(X0)[:, :-3].T)
        B = theta.sig_beta * np.diag(np.ones(len(X0)))
        C = theta.sig_zeta
        D = mk52(np.array(X0)[:, -3:-1], [theta.l1, theta.l2], theta.sig_m)
        Kx = A + B + C + D

        L = np.array([
            np.array([theta.rho[str(sorted([i, j]))] if i >= j else 0.0 for j in range(theta.n_IS)])
            for i in range(theta.n_IS)
        ])
        # Normalize L to stop over-scaling values small
        if theta.normalize_L:
            L = L / np.linalg.norm(L)
        # Force it to be positive semi-definite
        Ks = L.dot(L.T)
        if theta.normalize_Ks:
            Ks = Ks / np.linalg.norm(Ks)

        e = np.diag(np.array([theta.e1, theta.e2]))
        Ks = e.dot(Ks.dot(e))

        return np.kron(Ks, Kx)

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

    sim.theta.e1, sim.theta.bounds['e1'] = None, (1E-1, 1.0)
    sim.theta.e2, sim.theta.bounds['e2'] = None, (1E-1, 1.0)

    # # NOTE! This is a reserved keyword in misoKG.  We will generate a list of the same length
    # # of the information sources, and use this for scaling our IS.
    sim.theta.rho = {"[0, 0]": 1.0, "[0, 1]": 0.96, "[1, 1]": 1.0}
    sim.theta.bounds['rho [0, 0]'] = (0.1, 1.0)
    sim.theta.bounds['rho [0, 1]'] = (0.1, 1.0)
    sim.theta.bounds['rho [1, 1]'] = (0.1, 1.0)

    sim.theta.set_hp_names()

    sim.primary_rho_opt = False
    sim.update_hp_only_with_IS0 = False
    sim.update_hp_only_with_overlapped = False

    sim.theta.normalize_L = False
    sim.theta.normalize_Ks = False

    # This was a test feature that actually over-wrote rho to be PSD
    # sim.force_rho_psd = True
    sim.recommendation_kill_switch = "FAPbBrBrCl_THTO_0"

    ###################################################################################################

    # Start simulation
    sim.run()

