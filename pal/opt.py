'''
Multi-Information Source Optimization with a Knowledge Gradient (misoKG)

This module contains code to run misoKG for a list of information sources
(ranging from expensive to cheap as 0 to N).

**Classes**

    - misokg

**Functions**

    - x

**Examples**

...

'''

# Clancelot Dependencies
from squid import geometry
from squid.jobs import Job
from squid.print_helper import printProgressBar as ppb

# Python Standard Modules
import os
import copy
import time
import itertools
import numpy as np
import scipy.stats
import cPickle as pickle

# PAL imports
from pal.stats.MLE import MLE
from pal.stats.MAP import MAP
import pal.stats.bayesian as bayes
import pal.utils.strings as pal_strings
from pal.constants.solvents import solvents
from pal.stats.hyperparameters import Theta
from pal.acquisition.EI import getNextSample_EI
from pal.acquisition.kg import getNextSample_kg
from pal.acquisition.misokg import getNextSample_misokg
from pal.stats.likelihood import gaussian as gaussian_loglike
from pal.stats.likelihood import bonilla as bonilla_loglike

# LHS
from squid import doe_lhs


# Fake Job for debugging
class FakeJob(Job):
    def wait(self):
        pass

    def get_result(self):
        return -1.0


class Optimizer(object):
    '''
    '''
    def __init__(self, debug=False, numerical=False):
        # File names
        self.fname_out = "out.dat"
        self.fname_historical = None

        # Information sources, in order from expensive to cheap
        self.IS = []
        self.costs = None
        self.total_cost = 0.0
        self.best_vs_total_cost = []
        self.historical = None
        self.historical_nsample = 5
        self.n_start = 5  # The number of possible starting hyperparameters for MLE L-BFGS minimization.
        self.hyperparameter_objective = MLE  # The objective function for minimizing the hyperparameters
        self.prior = None
        self.reopt = None
        self.ramp_opt = None
        self.parallel = False
        self.update_hp_only_with_IS0 = False
        self.update_hp_only_with_overlapped = False
        self.force_rho_psd = False

        self.recommendation_kill_switch = None
        self.iteration_kill_switch = None
        self.cost_kill_switch = None

        self.noise = False
        self.loglike = gaussian_loglike
        self.dynamic_pc = False

        self.primary_rho_opt = False
        self.indices_overlap_len = 0
        self.indices_overlap_changed = True

        # Flag to solve an N-dim numerical problem instead of PAL
        self.numerical = numerical
        self.domain = None
        self.sample_n_from_domain = 1000

        if not numerical:
            # Possible compositions by default
            self.A = ["Cs", "MA", "FA"]
            self.B = ["Pb"]
            self.X = ["Cl", "Br", "I"]
            self.all_X = None
            self.solvents = copy.deepcopy(solvents)
            self.S = list(set([v["name"] for k, v in self.solvents.items()]))
            self.mixed_halides = True
            self.mixed_solvents = False
            self.combinations = None

        # Parameters for debugging and overwritting
        self.debug = debug
        self.verbose = False
        self.overwrite = True  # If True, warning, else Error
        self.sequential = True
        self.save_extra_files = False  # Additional output for post processing
        self.mu_fname = None
        self.sig_fname = None
        self.hp_fname = None
        self.logger_fname = None
        self.sample_fname = None
        self.combos_fname = None
        self.acquisition_fname = None

        # Functions
        self.mean = None
        self.cov = None
        self.theta = Theta()
        self.acquisition = getNextSample_EI

        # SETUPS FOR DEBUGS
        if self.debug:
            self.IS = [lambda x, y, z: FakeJob(1), lambda x, y, z: FakeJob(2)]

    def _get_info_source_map(self, X):
        return [
            [
                i for i in range(len(X)) if X[i][0] == j
            ]
            for j in range(int(np.array(X)[:, 0].max()) + 1)
        ]

    def get_combos_pure_solvent(self, A=None, B=None, X=None, S=None):
        '''
        This function will return a reduced list of unique possible ABX3 combinations, given
        a list of possible A-site, B-site, and X-site atoms (as well as solvent).  Note, this
        will only work for a pure solvent system.
        '''

        if A is None:
            A = self.A
        if B is None:
            B = self.B
        if X is None:
            X = self.X
        if S is None:
            S = self.S

        # Error handling
        for s, x in zip(["A", "B", "X", "S"], [A, B, X, S]):
            assert isinstance(x, list), "%s is not a list in get_combos_pure." % s

        if self.mixed_halides:
            combos = [
                "".join(list(x[:2]) + list(sorted(x[2:5])) + list(x[5:]))
                for x in itertools.product(
                    A, B, X, X, X, ["_"], S
                )
            ]
        else:
            combos = [
                "".join(list(x[:2]) + [x[2]] * 3 + list(x[3:]))
                for x in itertools.product(
                    A, B, X, ["_"], S
                )
            ]
        combos = sorted(geometry.reduce_list(combos))

        final_combos = []
        for i in range(len(self.IS)):
            for combo in combos:
                final_combos.append(combo + "_" + str(i))

        self.combinations = final_combos
        self.all_X = pal_strings.alphaToNum(self.combinations, solvents, mixed_halides=self.mixed_halides)
        self.all_solvent_properties = np.array(self.all_X)[:, -3:-1]
        self.all_Y = np.array([0 for i in range(len(self.all_X))])

        return final_combos

    def sample_numerical(self, allow_reduced=False, MAX_LOOP=50):
        '''
        Sample numerically from the input sample space stored in self.domain.
        NOTE! self.domain must be a list of tuples essentially indicating the range.
        self.n_from_domain will decide how the domain is divided.
        '''
        # Step 1 - Sample from the domain
        self.sampled_domain = doe_lhs.lhs(len(self.domain), self.sample_n_from_domain)
        # Step 2 - Scale to be in range
        self.sampled_domain = np.array([
            [lhs_x * (bnd[1] - bnd[0]) + bnd[0] for lhs_x, bnd in zip(lhs_sample, self.domain)]
            for lhs_sample in self.sampled_domain
        ])

        passed = False
        for counter in range(MAX_LOOP):
            # Step 3a - Grab the initial historical sample
            history = doe_lhs.lhs(len(self.domain), self.historical_nsample)
            # Step 3b - Find closest points to the sampled historical in self.sampled_domain
            indices = [
                sorted([(i, np.linalg.norm(h - sd)) for i, sd in enumerate(self.sampled_domain)],
                       key=lambda x: x[1])[0][0]
                for h in history
            ]
            # Step 3c - Ensure a unique list! And print a warning if historical was simplified
            # due to LHS sampling too many near points.
            sampled_indices = list(set(indices))
            passed = len(sampled_indices) == len(indices)
            if passed:
                break
    
            if len(sampled_indices) != len(indices):
                if allow_reduced:
                    print("Warning - Will sample from a reduced space of %d instead of %d." % (len(sampled_indices), len(indices)))
        if not passed:
            raise Exception("Error - Unable to sample from space without duplicates.")

        # Step 4 - Get historical now
        historical = [self.sampled_domain[i] for i in sampled_indices]

        # Step 5 - Sample IS
        self.historical = []
        self.all_X = []
        self.sampled_indices = []
        for i, IS in enumerate(self.IS):
            for j, hist in zip(sampled_indices, historical):
                self.historical.append([i] + list(hist) + [IS(*hist)])
                self.sampled_indices.append(j + i * self.sample_n_from_domain)
            for x in self.sampled_domain:
                self.all_X.append([i] + list(x))
        self.all_X = np.array(self.all_X)
        self.historical_nsample = len(self.historical)
        self.historical = np.array(self.historical)
        self.sampled_X = self.historical[:, :-1].tolist()
        self.sampled_objectives = self.historical[:, -1].tolist()
        self.all_Y = np.zeros(len(self.all_X))
        def pretty_str(x):
            return "[" + str(x[0]) + ", " + ', '.join(["%.4f" % v for v in x[1:]]) + "]"
        self.combinations = map(lambda x: pretty_str(x), self.all_X)
        self.sampled_names = [pretty_str(self.all_X[x, :]) for x in self.sampled_indices]

    def sample(self, specify=None, debug=False, MAX_LOOP=10, allow_reduced=False):
        '''
        This function will run, in parallel, N_samples of the objective functions for
        historical data generation.  Note, these are run for EVERY information source.
        '''

        if debug:
            print("Collecting LHS samples...")

        if specify is None:
            counter, samples = 0, []
            while len(samples) != self.historical_nsample and counter < MAX_LOOP:

                # Grab a latin hypercube sample
                samples = doe_lhs.lhs(int(self.mixed_halides) * 2 + 2, self.historical_nsample)
                # Round the LHS and figure out the samples
                solvent_ranges = [i * 1.0 / len(self.S) for i in range(1, len(self.solvents) + 1)]
                solv = lambda v: self.S[[v <= s for s in solvent_ranges].index(True)]
                trio = lambda v: [int(v > (chk - 1.0 / 3.0) and v <= chk) for chk in [1. / 3., 2. / 3., 1.0]]

                # Grab our samples
                if self.mixed_halides:
                    halides = [sorted([s[0], s[1], s[2]]) for s in samples]
                    halides = [[trio(h) for h in hh] for hh in halides]
                    samples = [
                        h[0] + h[1] + h[2] + trio(s[3]) +
                        [self.solvents[solv(s[-1])]["density"],
                         self.solvents[solv(s[-1])]["dielectric"],
                         self.solvents[solv(s[-1])]["index"]]
                        for h, s in zip(halides, samples)]
                else:
                    samples = [
                        trio(s[0]) + trio(s[1]) +
                        [self.solvents[solv(s[-1])]["density"],
                         self.solvents[solv(s[-1])]["dielectric"],
                         self.solvents[solv(s[-1])]["index"]]
                        for s in samples]

                # Ensure no duplicates
                samples = sorted(samples, key=lambda x: x[-1])
                samples = [tuple(s) for s in samples]
                samples = [list(s) for s in set(samples)]

                counter += 1
        else:
            if isinstance(specify, int):
                specify = [specify]
            self.historical_nsample = len(specify)
            samples = [self.combinations[i] for i in specify]
            samples = pal_strings.alphaToNum(
                samples,
                solvents,
                mixed_halides=self.mixed_halides,
                name_has_IS=True)
            samples = [s[1:] for s in samples]  # Remove the IS label from the descriptor

        if allow_reduced:
            print("Warning - Will sample from subspace due to duplicates (%d instead of %d)." % (len(samples), self.historical_nsample))
            self.historical_nsample = len(samples)
        elif specify is None:
            assert counter < MAX_LOOP, "Error - Unable to sample from space without duplicates!"

        if debug:
            print("Will sample %s" % str(samples))

        # Now, run these simulations to get the sample points
        jobs = []
        for i, sample in enumerate(samples):
            if debug:
                print "Running %s..." % sample
            s = pal_strings.parseNum(sample, self.solvents, mixed_halides=self.mixed_halides, num_has_IS=False)
            hat, cat, _, solv, _ = pal_strings.parseName(s, name_has_IS=False)
            cat = cat[0]
            if not self.mixed_halides:
                hat = hat[0]
            if debug:
                print("\tAdding %s to sample runs..." % s)

            for j, obj in enumerate(self.IS):
                jobs.append([[j] + copy.deepcopy(sample), obj(hat, cat, solv)])

        # Now, get results from each simulation
        samples = []
        for sample, j in jobs:
            if not isinstance(j, float):
                j.wait()
                samples.append(sample + [j.get_result()])
            # In special situations, when we are reading from a list for example, we don't need to worry
            # about a job object, and can just assign the value directly.
            else:
                samples.append(sample + [j])

            s = pal_strings.parseNum(samples[-1][:-1], self.solvents, mixed_halides=self.mixed_halides, num_has_IS=True)
            if debug:
                print("\t%s was found as %lg" % (s, samples[-1][-1]))

        # Save the sampled data
        fptr = open(self.fname_historical, "w")
        pickle.dump(samples, fptr)
        fptr.close()
        self.historical = samples

        if debug:
            print("Done Collecting Samples\n")

    def assign_samples(self):
        self.sampled_X = [v[:-1] for v in self.historical]
        self.sampled_names = [
            pal_strings.parseNum(v, self.solvents, mixed_halides=self.mixed_halides, sort=True, num_has_IS=True)
            for v in self.sampled_X
        ]
        self.sampled_indices = [
            self.combinations.index(v) for v in self.sampled_names
        ]

        assert len(self.sampled_indices) == len(list(set(self.sampled_indices))), "Error - Sampled indices contain duplicates!"

        self.sampled_solvent_properties = np.array([np.array(v[-4:-2]) for v in self.historical])
        self.sampled_objectives = [v[-1] for v in self.historical]

    def save(self):
        if self.save_extra_files:
            for i in range(len(self.IS)):
                indices = self._get_info_source_map(self.all_X)[i]

                if self.mu_fname is not None:
                    mu = np.array(self.mu)[indices]
                    fptr = open(self.mu_fname + "_IS" + str(i), 'a')
                    fptr.write('\t'.join(["%.2f" % m for m in mu]) + "\n")
                    fptr.close()

                if self.sig_fname is not None:
                    K = np.array(self.K)[indices, indices]
                    fptr = open(self.sig_fname + "_IS" + str(i), 'a')
                    fptr.write('\t'.join(["%.2f" % sig for sig in K]) + "\n")
                    fptr.close()

            if self.hp_fname is not None:
                fptr = open(self.hp_fname, 'a')
                fptr.write(str(self.theta))
                fptr.close()

    def updateHPs(self, use_theta=True):
        '''
        This function updates the hyperparameters with user specifications.
        '''

        # If we are in the edge case where no HPs need optimizing
        # (they have all been set by the user prior), then stop.
        if self.theta.items() == []:
            return None

        # Get a list of indices pointed to IS0
        indices_IS0 = self._get_info_source_map(self.sampled_X)[0]

        indices_to_use = range(len(self.sampled_X))
        if self.update_hp_only_with_IS0:
            indices_to_use = indices_IS0

        if self.update_hp_only_with_overlapped:
            if not self.indices_overlap_changed:
                return None
            indices_to_use = self.indices_overlap

        # If we are to dynamically calculate rho based on pearson correlation
        # coefficients, do so here.
        if self.dynamic_pc:
            overlapped_x = np.array(self.sampled_X)[self.indices_overlap]
            overlapped_y = np.array(self.sampled_objectives)[self.indices_overlap]
            split_by_is_dict = [{str(x[1:]): y for x, y in zip(overlapped_x, overlapped_y) if x[0] == j} for j in range(len(self.IS))]
            keys = split_by_is_dict[0].keys()
            split_by_is = np.array([[y for y in [x[k] for x in split_by_is_dict]] for k in keys])
            for i in range(len(self.IS)):
                for j in range(i + 1, len(self.IS)):
                    k = str(sorted([i, j]))
                    self.theta.rho[k] = scipy.stats.pearsonr(split_by_is[:, i], split_by_is[:, j])[0]
                    if self.theta.rho[k] == 1.0:
                        self.theta.rho[k] = 1.0 - 1E-6

        # If we want to optimize rho in which only overlapped indices (samples
        # have been taken at all available IS), then do so.
        if self.primary_rho_opt:
            if self.indices_overlap_changed:
                self.theta = self.hyperparameter_objective(
                    np.array(self.sampled_X)[self.indices_overlap],
                    np.array(self.sampled_objectives)[self.indices_overlap],
                    self.mean,
                    self.cov,
                    self.theta,
                    self.prior,
                    n_start=self.n_start,
                    parallel=self.parallel,
                    force_rho_psd=self.force_rho_psd,
                    loglike=self.loglike
                )
            # When doing the primary_rho_opt, we freeze rho
            for r in self.theta.hp_names:
                if r.startswith("rho"):
                    self.theta.freeze(r)

        # Optimize the hyperparameters appropriately
        self.theta = self.hyperparameter_objective(
            np.array(self.sampled_X)[indices_to_use],
            np.array(self.sampled_objectives)[indices_to_use],
            self.mean,
            self.cov,
            self.theta,
            self.prior,
            n_start=self.n_start,
            parallel=self.parallel,
            use_theta=use_theta,
            force_rho_psd=self.force_rho_psd,
            loglike=self.loglike
        )

        # When doing primary_rho_opt, we had frozen rho.  Unfreeze it here.
        if self.primary_rho_opt:
            for r in self.theta.hp_names:
                if r.startswith("rho"):
                    self.theta.unfreeze(r)


    def updatePosterior(self, point=None):
        '''
        Update the posterior.

        If we want to try out using the Bonilla approach, where the
        covariance matrix is a kronecker product of Ks and Kx, then we
        remove redundant calculations by only taking a single representation
        of the X domain.  This is seen in the get_reduced_X boolean that is
        defined.

        **Parameters**

            point: *tuple, int, float*
                A point to update the posterior with. If not specified, all sampled indices and objectives are used.
        '''
        get_reduced_X = len(self.IS) > 0
        X0 = self.all_X

        if get_reduced_X:
            # Get a list of all unique X, removing initial IS identifier
            X0 = []
            for x in self.all_X:
                if not any([all([a == b for a, b in zip(x[1:], xchk)]) for xchk in X0]):
                    X0.append(x[1:])
            # Now, we get the sub-covariance matrix for the specified sampled X and Y
            indices = []
            for l in range(len(self.IS)):
                for i, x in enumerate(X0):
                    test = [l] + list(x)
                    if any([all([a == b for a, b in zip(test, xchk)]) for xchk in self.all_X]):
                        indices.append(l * len(X0) + i)
 
        if self.sequential:
            if point is not None:
                self.mu, self.K = bayes.sequential_posterior_update(point[0], point[1], self.mu, self.K)
            else:
                self.mu = self.mean(self.all_X, self.all_Y, self.theta)
                self.K = self.cov(X0, self.all_Y, self.theta)

                if get_reduced_X:
                    self.K = self.K[np.ix_(indices, indices)]
                for x, y in zip(self.sampled_indices, self.sampled_objectives):
                    self.mu, self.K = bayes.sequential_posterior_update(x, y, self.mu, self.K)
        else:
            raise Exception("HAVE NOT IMPLEMENTED FOR NON-SEQUENTIAL BAYESIAN OPT!")
        self.all_Y = self.mu

        assert not np.any(np.isnan(self.K)), "Error - covariance matrix has nan in it!"
        assert not np.any(np.isnan(self.mu)), "Error - predicted mean has nan in it!"

    def run(self):
        '''
        Run misoKG.
        '''

        if self.save_extra_files and self.overwrite:
            os.system("rm %s %s" % (self.mu_fname, self.sig_fname))

        # Error Handling
        assert self.costs is not None, "Error - You must specify costs before running!"
        assert len(self.costs) == len(self.IS), "Error - You must specify the same number of information sources as costs!"

        if self.verbose:
            print("\n-------------------------------------------------------------------------------------")
            print("Beginning optimization.")
            # print("\tParallel = %s" % str(self.parallel))
            print("    Number of Information Sources = %d" % len(self.IS))
            print("    Acquisition = "),
            if self.acquisition == getNextSample_misokg:
                print("misoKG with a cost list = %s" % str(self.costs))
            elif self.acquisition == getNextSample_EI:
                print("EI")
            elif self.acquisition == getNextSample_kg:
                print("KG")
            else:
                print("Custom!")
            if self.hyperparameter_objective == MLE:
                obj_name = "MLE"
            elif self.hyperparameter_objective == MAP:
                obj_name = "MAP"
            else:
                obj_name = "Custom"
            if self.loglike == gaussian_loglike:
                loglike_name = "Gaussian"
            elif self.loglike == bonilla_loglike:
                loglike_name = "Bonilla"
            else:
                loglike_name = "Custom"
            print("The Hyperparameter Objective is %s with %d starting samples." % (obj_name, self.n_start))
            print("The loglikelihood method is %s." % loglike_name)
            if self.dynamic_pc:
                print("Will use a dynamic pearson correlation coefficient for rho.")
            print("Will optimize the following parameters:")
            print("    " + ', '.join(self.theta.hp_names))
            print("-------------------------------------------------------------------------------------")
        # Start - TIMER
        self.t0 = time.time()

        # Step 1 - Ensure we have our historical training set.  If not, then
        # generate one.
        if self.fname_historical is None:
            self.fname_historical = "historical.dat"
            if self.numerical:
                self.sample_numerical()
            else:
                self.sample()
        else:
            self.historical = pickle.load(open(self.fname_historical, 'r'))
            if self.numerical:
                if len(self.historical[0]) != len(self.domain) + 1:
                    raise Exception("The historical data seems to be incorrect for misoKG.  Maybe the IS associated with each point was not included?")
            if len(self.historical[0]) not in [10, 17]:
                raise Exception("The historical data seems to be incorrect for misoKG.  Maybe the IS associated with each point was not included?")

        # Step 2 - Generate a full list of our sample space if it has not been given
        if self.mixed_solvents and not self.numerical:
            raise Exception("Mixed Solvents have not been implemented properly.")
        else:
            if self.combinations is None and not self.numerical:
                self.combinations = self.get_combos_pure_solvent()
        if self.all_X is None and not self.numerical:
            self.all_X = pal_strings.alphaToNum(self.combinations, solvents, mixed_halides=self.mixed_halides)
            self.all_solvent_properties = np.array(self.all_X)[:, -3:-1]
            self.all_Y = np.array([0 for i in range(len(self.all_X))])

        # Step 2.5 - Store our X and Y points
        if not self.numerical:
            self.assign_samples()
        # Store a list of samples that have been sampled at all information sources
        self.indices_overlap = range(len(self.sampled_X))

        # Step 3 - Get our hyperparameters.  As we don't have initial ones,
        # don't use_theta for this instance.
        self.updateHPs(use_theta=False)

        # Step 3.5 - Initialize indices_overlap variables
        self.indices_overlap_len = len(self.indices_overlap)
        self.indices_overlap_changed = False

        # Step 4 - Update the posterior based on the historical data.
        self.updatePosterior()

        if not self.numerical:
            # Save combinations and default save actions
            if self.combos_fname is not None:
                fptr = open(self.combos_fname, 'w')
                for i, c in enumerate(self.combinations):
                    fptr.write("%d\t%s\n" % (i, c))
                fptr.close()
            self.save()
    
            # Step 5 - Begin the main loop
            start, stop = len(self.sampled_X), len(self.combinations)
        else:
            start, stop = len(self.sampled_X), len(self.all_X)

        best_found_in = start
        best_value = max(np.array(self.sampled_objectives)[self._get_info_source_map(self.sampled_X)[0]])
        best_index = self.sampled_indices[self.sampled_objectives.index(best_value)]
        best_name = self.combinations[best_index]

        # Initialize our costs based on the sampled so far
        self.total_cost = sum([self.costs[int(x[0])] for x in self.sampled_X])

        best_prediction = max(np.array(self.mu)[self._get_info_source_map(self.all_X)[0]])
        best_prediction = list(np.array(self.mu)[self._get_info_source_map(self.all_X)[0]]).index(best_prediction)
        best_prediction = self._get_info_source_map(self.all_X)[0][best_prediction]
        recommendation = self.combinations[best_prediction]

        if self.save_extra_files and self.sample_fname is not None:
            fptr = open(self.sample_fname, 'a')
            for v in zip(self.sampled_names, self.sampled_objectives):
                fptr.write("%s\t%.4f\n" % v)

        # Begin the main loop
        fully_sampled = False
        recommendation_kill_flag = False
        iteration_kill_flag = False
        cost_kill_flag = False
        for index in range(start, stop):
            if self.iteration_kill_switch is not None and index >= self.iteration_kill_switch:
                iteration_kill_flag = True
                break

            # If we have sampled all the IS0, and noise doesn't exist, then we gracefully exit
            if not self.noise and all([i_IS0 in self.sampled_indices for i_IS0 in self._get_info_source_map(self.all_X)[0]]):
                fully_sampled = True
                break

            # Step 6 - acquisition Function.  Decide on next point(s) to sample.
            next_point = self.acquisition(
                self.mu,
                #self.theta.rho_matrix(self.all_X) * self.K,
                self.K,
                max(self.sampled_objectives),
                len(self.combinations),
                self.costs,
                self.all_X,
                self.sampled_indices,
                save=self.acquisition_fname
            )
            if next_point in self.sampled_indices:
                print("\nFAILURE!!!! SAMPLED # %s - Index = %d# POINT TWICE!\n" % (self.combinations[next_point], next_point))
                print("K Diagonal = %s" % ' '.join(["%f" % v for v in np.diag(self.K)]))
                print("K[%d] = %s" % (next_point, ' '.join(["%f" % v for v in self.K[next_point]])))
                print("Sampled Points = %s" % str(self.sampled_indices))
                raise Exception("Error - acquisition function grabbed an already sampled point!")

            if self.verbose:
                r = -1.23
                if "[0, 1]" in self.theta.rho:
                    r = self.theta.rho["[0, 1]"]
                suffix = "(iter %d) %s = %.4f, sampling %s. Recommendation = %s, Current Cost = %.2f, Rho = %.3f" % (
                    best_found_in, best_name, best_value, self.combinations[next_point], recommendation, self.total_cost, r
                )
                ppb(index,
                    stop,
                    prefix='Running',
                    suffix=suffix,
                    pad=True)
                if self.logger_fname is not None:
                    fptr = open(self.logger_fname, 'a')
                    fptr.write(suffix + "\n")
                    fptr.close()
            if self.recommendation_kill_switch is not None and recommendation == self.recommendation_kill_switch:
                recommendation_kill_flag = True
                break

            # Step 7 - Sample point(s)
            self.sampled_indices.append(next_point)
            self.sampled_names.append(self.combinations[next_point])
            if not self.numerical:
                self.sampled_X.append(pal_strings.alphaToNum(self.sampled_names[-1], solvents, mixed_halides=self.mixed_halides)[0])
                h, c, _, s, info_lvl = pal_strings.parseName(self.sampled_names[-1])
                self.sampled_objectives.append(self.IS[info_lvl](h, c[0], s))
            else:
                x = self.all_X[next_point]
                self.sampled_X.append(x)
                info_lvl = int(x[0])
                self.sampled_objectives.append(self.IS[info_lvl](*x[1:]))

            if self.save_extra_files and self.sample_fname is not None:
                fptr = open(self.sample_fname, 'a')
                fptr.write("%s\t%.4f\n" % (self.sampled_names[-1], self.sampled_objectives[-1]))

            # Ensure we get an array of all sampled indices that have been sampled
            # at ALL information source levels
            chk = self.sampled_X[-1][1:]
            if self.numerical:
                found = [i for i, v in enumerate(self.sampled_X) if all(chk == v[1:])]
            else:
                found = [i for i, v in enumerate(self.sampled_X) if chk == v[1:]]
            # Assume we have 4 IS.  If we find 4 of chk, then we now have fully sampled chk.
            if len(found) == len(self.IS):
                for f in found:
                    if f not in self.indices_overlap:
                        self.indices_overlap.append(f)
                self.indices_overlap_changed = self.indices_overlap_len != len(self.indices_overlap)
                self.indices_overlap_len = len(self.indices_overlap)

            # Step 7.5 - Maybe re-opt the hyperparameters
            # Note, we do this in a two step approach.  First, we optimize all HPs based on
            # only data points that exist at all levels of theory.  Then we optimize
            # only at the highest level of theory sampled so far (IS0).
            if index != start and (self.reopt is not None and index % self.reopt == 0) or (self.ramp_opt is not None and index < self.ramp_opt):
                self.updateHPs()
                # Step 8a - Update the posterior completely if we are reoptimizing the HPs
                self.updatePosterior()
            else:
                # Step 8b - Update the posterior with only the newest sampled point
                self.updatePosterior((self.sampled_indices[-1], self.sampled_objectives[-1]))

            self.save()

            # Count the cost of this iteration
            self.total_cost += self.costs[info_lvl]

            if self.cost_kill_switch is not None and self.total_cost > self.cost_kill_switch:
                cost_kill_flag = True
                break

            # Get our recommendation from max(mu) for only IS0
            best_prediction = max(np.array(self.mu)[self._get_info_source_map(self.all_X)[0]])
            best_prediction = list(np.array(self.mu)[self._get_info_source_map(self.all_X)[0]]).index(best_prediction)
            best_prediction = self._get_info_source_map(self.all_X)[0][best_prediction]
            recommendation = self.combinations[best_prediction]

            # Get the best sampled so far
            potential_best = max(np.array(self.sampled_objectives)[self._get_info_source_map(self.sampled_X)[0]])
            if potential_best > best_value:
                best_found_in = index
                best_value = potential_best
                best_index = self.sampled_indices[self.sampled_objectives.index(best_value)]
                best_name = self.combinations[best_index]

        # END TIMER
        self.t1 = time.time()

        if self.verbose:
            print("-----------------------")
            print("PAL Optimizer has completed in %.2f s" % (self.t1 - self.t0))
            if fully_sampled:
                print("Optimizer quit early as IS0 was fully sampled")
            if recommendation_kill_flag:
                print("Optimizer quit early due to recommendation of %s." % self.recommendation_kill_switch)
            if iteration_kill_flag:
                print("Optimizer quit early due to exceeding %d iterations." % self.iteration_kill_switch)
            if cost_kill_flag:
                print("Optimizer quit early due to exceeding %.4f cost." % self.cost_kill_switch)
            print("-----------------------")
            print("Best combination: %s" % best_name)
            print("       Objective: %.4f" % best_value)
            print("       Maximized: %d" % best_found_in)
            print("-----------------------")
            print self.theta
            print("-------------------------------------------------------------------------------------\n")


def _test_get_combos_pure_solvent():
    obj = Optimizer(debug=True)
    combos = obj.get_combos_pure_solvent(["Cs", "MA", "FA"], ["Pb"], ["Cl", "Br", "I"], ["ACE", "NM"])
    return combos == ['CsPbBrBrBr_ACE_0', 'CsPbBrBrBr_NM_0', 'CsPbBrBrCl_ACE_0', 'CsPbBrBrCl_NM_0',
                      'CsPbBrBrI_ACE_0', 'CsPbBrBrI_NM_0', 'CsPbBrClCl_ACE_0', 'CsPbBrClCl_NM_0',
                      'CsPbBrClI_ACE_0', 'CsPbBrClI_NM_0', 'CsPbBrII_ACE_0', 'CsPbBrII_NM_0',
                      'CsPbClClCl_ACE_0', 'CsPbClClCl_NM_0', 'CsPbClClI_ACE_0', 'CsPbClClI_NM_0',
                      'CsPbClII_ACE_0', 'CsPbClII_NM_0', 'CsPbIII_ACE_0', 'CsPbIII_NM_0',
                      'FAPbBrBrBr_ACE_0', 'FAPbBrBrBr_NM_0', 'FAPbBrBrCl_ACE_0', 'FAPbBrBrCl_NM_0',
                      'FAPbBrBrI_ACE_0', 'FAPbBrBrI_NM_0', 'FAPbBrClCl_ACE_0', 'FAPbBrClCl_NM_0',
                      'FAPbBrClI_ACE_0', 'FAPbBrClI_NM_0', 'FAPbBrII_ACE_0', 'FAPbBrII_NM_0',
                      'FAPbClClCl_ACE_0', 'FAPbClClCl_NM_0', 'FAPbClClI_ACE_0', 'FAPbClClI_NM_0',
                      'FAPbClII_ACE_0', 'FAPbClII_NM_0', 'FAPbIII_ACE_0', 'FAPbIII_NM_0',
                      'MAPbBrBrBr_ACE_0', 'MAPbBrBrBr_NM_0', 'MAPbBrBrCl_ACE_0', 'MAPbBrBrCl_NM_0',
                      'MAPbBrBrI_ACE_0', 'MAPbBrBrI_NM_0', 'MAPbBrClCl_ACE_0', 'MAPbBrClCl_NM_0',
                      'MAPbBrClI_ACE_0', 'MAPbBrClI_NM_0', 'MAPbBrII_ACE_0', 'MAPbBrII_NM_0',
                      'MAPbClClCl_ACE_0', 'MAPbClClCl_NM_0', 'MAPbClClI_ACE_0', 'MAPbClClI_NM_0',
                      'MAPbClII_ACE_0', 'MAPbClII_NM_0', 'MAPbIII_ACE_0', 'MAPbIII_NM_0', 'CsPbBrBrBr_ACE_1',
                      'CsPbBrBrBr_NM_1', 'CsPbBrBrCl_ACE_1', 'CsPbBrBrCl_NM_1', 'CsPbBrBrI_ACE_1',
                      'CsPbBrBrI_NM_1', 'CsPbBrClCl_ACE_1', 'CsPbBrClCl_NM_1', 'CsPbBrClI_ACE_1',
                      'CsPbBrClI_NM_1', 'CsPbBrII_ACE_1', 'CsPbBrII_NM_1', 'CsPbClClCl_ACE_1',
                      'CsPbClClCl_NM_1', 'CsPbClClI_ACE_1', 'CsPbClClI_NM_1', 'CsPbClII_ACE_1',
                      'CsPbClII_NM_1', 'CsPbIII_ACE_1', 'CsPbIII_NM_1', 'FAPbBrBrBr_ACE_1', 'FAPbBrBrBr_NM_1',
                      'FAPbBrBrCl_ACE_1', 'FAPbBrBrCl_NM_1', 'FAPbBrBrI_ACE_1', 'FAPbBrBrI_NM_1',
                      'FAPbBrClCl_ACE_1', 'FAPbBrClCl_NM_1', 'FAPbBrClI_ACE_1', 'FAPbBrClI_NM_1',
                      'FAPbBrII_ACE_1', 'FAPbBrII_NM_1', 'FAPbClClCl_ACE_1', 'FAPbClClCl_NM_1', 'FAPbClClI_ACE_1',
                      'FAPbClClI_NM_1', 'FAPbClII_ACE_1', 'FAPbClII_NM_1', 'FAPbIII_ACE_1', 'FAPbIII_NM_1',
                      'MAPbBrBrBr_ACE_1', 'MAPbBrBrBr_NM_1', 'MAPbBrBrCl_ACE_1', 'MAPbBrBrCl_NM_1',
                      'MAPbBrBrI_ACE_1', 'MAPbBrBrI_NM_1', 'MAPbBrClCl_ACE_1', 'MAPbBrClCl_NM_1',
                      'MAPbBrClI_ACE_1', 'MAPbBrClI_NM_1', 'MAPbBrII_ACE_1', 'MAPbBrII_NM_1', 'MAPbClClCl_ACE_1',
                      'MAPbClClCl_NM_1', 'MAPbClClI_ACE_1', 'MAPbClClI_NM_1', 'MAPbClII_ACE_1', 'MAPbClII_NM_1',
                      'MAPbIII_ACE_1', 'MAPbIII_NM_1']


def run_unit_tests():
    assert _test_get_combos_pure_solvent(), "pal.opt.get_combos_pure_solvent() failed."

