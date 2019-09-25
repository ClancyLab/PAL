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


def run_misokg(run_index, sffx="misokg", SAMPLE_DOMAIN=1000):

    FOLDER = "RNS%d" % SAMPLE_DOMAIN
 
    scaled = False
    dpc = False
    invert_dpc = False
    scaled = False
    use_I = False
    use_J = False
    use_miso = False
    
    if sffx == "misokg":
        use_miso = True
    elif sffx == "bdpc":
        dpc = True
    elif sffx == "bidpc":
        dpc = True
        invert_dpc = True
    elif sffx == "bvl":
        pass
    elif sffx == "bsvl":
        scaled = True
    elif sffx == "bI":
        use_I = True
    elif sffx == "bu":
        use_J = True
    else:
        raise Exception("This sffx (%s) is not accounted for." % sffx)

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
        lambda x1, x2: -1.0 * rosenbrock(x1, x2),
        lambda x1, x2: -1.0 * (rosenbrock(x1, x2) + 0.1 * np.sin(10.0 * x1 + 5.0 * x2))
    ]
    #sim.IS = [
    #    lambda x1, x2: (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3 + np.random.normal()
    #    lambda x1, x2: (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3 + 2.0 * np.sin(10.0 * x1 + 5.0 * x2)
    #]
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
    sim.mean = lambda X, Y, theta: np.array([-456.3 for _ in Y])

    def cov_miso(X0, Y, theta):
        Kx = squared(np.array(X0)[:, 1:], [theta.l1, theta.l2], theta.sig_1)
        Kx_l = squared(np.array(X0)[:, 1:], [theta.l3, theta.l4], theta.sig_2)
        return np.block([[Kx, Kx], [Kx, Kx + Kx_l]])

    def cov_bonilla(X0, Y, theta):
        Kx = squared(np.array(X0)[:, 1:], [theta.l1, theta.l2], theta.sig_1)
        Kx = Kx + 1E-6 * np.eye(Kx.shape[0])

        if use_J:
            Ks = np.ones((theta.n_IS, theta.n_IS)) * (1.0 - 1E-6) + np.eye(theta.n_IS) * 1E-6
        elif use_I:
            Ks = np.eye(theta.n_IS)
        elif dpc and invert_dpc:
            Ks = np.array([
                np.array([1.0 if i != j else theta.rho["[0, %d]" % i]**(-2.0) for j in range(theta.n_IS)])
                for i in range(theta.n_IS)
            ])
        elif dpc:
            Ks = np.array([
                np.array([theta.rho[str(sorted([i, j]))] for j in range(theta.n_IS)])
                for i in range(theta.n_IS)
            ])
        else:
            L = np.array([
                np.array([theta.rho[str(sorted([i, j]))] if i >= j else 0.0 for j in range(theta.n_IS)])
                for i in range(theta.n_IS)
            ])
            # Force it to be positive semi-definite
            Ks = L.dot(L.T)
            if theta.n_IS == 2:
                e = np.diag(np.array([theta.e1, theta.e2]))
            elif theta.n_IS == 3:
                e = np.diag(np.array([theta.e1, theta.e2, theta.e3]))
            else:
                raise Exception("HOW?")
            Ks = e.dot(Ks.dot(e))

        return np.kron(Ks, Kx)

    sim.theta.bounds = {}
    sim.theta.sig_1, sim.theta.bounds['sig_1'] = None, (1E-2, lambda _, Y: np.var(Y))
    sim.theta.l1, sim.theta.bounds['l1'] = None, (1E-1, 1)
    sim.theta.l2, sim.theta.bounds['l2'] = None, (1E-1, 1)

    if use_miso:
        sim.cov = cov_miso
        sim.theta.sig_2, sim.theta.bounds['sig_2'] = None, (1E-2, lambda _, Y: np.var(Y))
        sim.theta.l3, sim.theta.bounds['l3'] = None, (1E-1, 1)
        sim.theta.l4, sim.theta.bounds['l4'] = None, (1E-1, 1)
        sim.theta.rho = {str(sorted([i, j])): 1.0 for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
    else:
        sim.cov = cov_bonilla
 
        if scaled:
            sim.theta.e1, sim.theta.bounds['e1'] = None, (1E-1, 1.0)
            sim.theta.e2, sim.theta.bounds['e2'] = None, (1E-1, 1.0)
        else:
            sim.theta.e1, sim.theta.bounds['e1'] = 1.0, (1E-1, 1.0)
            sim.theta.e2, sim.theta.bounds['e2'] = 1.0, (1E-1, 1.0)

        sim.theta.rho = {"[0, 0]": None, "[0, 1]": None, "[1, 1]": None}
        if dpc or use_I or use_J:
            sim.theta.rho = {str(sorted([i, j])): 1.0 for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
            sim.dynamic_pc = dpc

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
    #sim.cost_kill_switch = sim.iteration_kill_switch * sim.costs[0]
    sim.run()

