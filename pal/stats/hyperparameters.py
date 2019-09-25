import copy
import numpy as np
from squid import structures


class Theta(structures.Struct):

    hp_names = None
    save_bounds = {}
    normalize_L = False
    normalize_Ks = False

    def set_hp_names(self):
        '''
        This function assigns the variable hp_names so that it is consistent.
        That is, Theta works as a dictionary, and order is not maintained.
        By setting hp_names, it will ensure that the order is maintained.
        '''
        if self.hp_names is None:
            self.hp_names = [k for k in self.__dict__.keys() if k not in ["rho", "bounds"] and self.__dict__[k] is None]
            if "rho" in self.__dict__.keys():
                reduced_rho_keys, _ = self.reduced_rho()
                for k in reduced_rho_keys:
                    if self.__dict__["rho"][k] is None:
                        self.hp_names.append("rho %s" % k)
                self.n_IS = int(max([eval(r)[0] for r in self.__dict__["rho"].keys()])) + 1

    def items(self):
        self.set_hp_names()
        return self.hp_names

    def assign(self, prop, value):
        self.set_hp_names()
        assert prop in self.hp_names, "%s is not in Theta!" % prop

        if prop.startswith("rho") and len(prop) > 3:
            self.__dict__["rho"][prop.split("rho")[1].strip()] = value
        else:
            self.__dict__[prop] = value

    def get(self, prop):
        self.set_hp_names()
        assert prop in self.hp_names, "%s not in Theta!" % prop

        if prop.startswith("rho") and len(prop) > 3:
            return self.__dict__["rho"][prop.split("rho")[1].strip()]
        else:
            return self.__dict__[prop]

    def wrap(self, hps):
        '''
        hps: *dict*
            A dictionary of hyperparameters.
        '''
        self.set_hp_names()
        # Handle updating rho differently
        for k, v in hps.items():
            if k.strip().startswith("rho "):
                self.__dict__["rho"][k.split("rho")[1].strip()] = hps[k]
            else:
                self.__dict__[k] = v

    def unwrap(self):
        self.set_hp_names()
        return [self.__dict__[k] if not k.strip().startswith("rho") else self.__dict__["rho"][k.split("rho")[1].strip()] for k in self.hp_names]

    def reduced_rho(self):
        if "rho" not in self.__dict__:
            return None
        #chk = lambda x: eval(x)[0] != eval(x)[1]
        reduced_rho_keys = list(set([str(sorted(eval(k))) for k in self.__dict__["rho"].keys()]))
        reduced_rho_keys.sort()
        reduced_rho_vals = [self.__dict__["rho"][k] for k in reduced_rho_keys]

        return reduced_rho_keys, reduced_rho_vals

    def freeze(self, prop):
        prop_id = prop
        if prop not in self.__dict__.keys() and prop.startswith("rho"):
            prop_id = prop.split("rho")[1].strip()
            assert prop_id in self.__dict__["rho"], "Error - %s is not in theta.rho (keys = %s)!" % (prop_id, ', '.join(self.__dict__["rho"].keys()))
        else:
            assert prop_id in self.__dict__, "Error - %s is not in theta (keys = %s)!" % (prop_id, ', '.join(self.__dict__.keys()))

        assert prop in self.__dict__["bounds"], "Error - %s is not in bounds (keys = %s)!" % (prop, ', '.join(self.__dict__["bounds"].keys()))

        if not prop.startswith("rho"):
            self.save_bounds[prop] = self.__dict__["bounds"][prop]
            self.__dict__["bounds"][prop] = (self.__dict__[prop], self.__dict__[prop])
        else:
            self.save_bounds[prop] = self.__dict__["bounds"][prop]
            self.__dict__["bounds"][prop] = (self.__dict__["rho"][prop_id], self.__dict__["rho"][prop_id])

    def unfreeze(self, prop):
        self.__dict__["bounds"][prop] = self.save_bounds[prop]

    def gen_psd_rho(self):
        # Generate a PSD rho factor within theta
        L = np.array([
            [self.__dict__["rho"][str(sorted([row, col]))] if row >= col else 0.0
             for col in range(self.n_IS)]
            for row in range(self.n_IS)
        ])
        if self.normalize_L:
            L = L / np.linalg.norm(L)
        rho_PSD = L.dot(L.T)
        if self.normalize_Ks:
            rho_PSD = rho_PSD / np.linalg.norm(rho_PSD)

        for k in self.__dict__["rho"].keys():
            row, col = eval(k)
            self.__dict__["rho"][k] = rho_PSD[row][col]

    def rho_matrix(self, X):
        return np.array(
            [np.array(
                [self.rho[str(sorted([a, b]))]
                 for a in np.array(X, dtype=int)[:, 0]])
                for b in np.array(X, dtype=int)[:, 0]
             ])

    def __str__(self):
        self.set_hp_names()
        return "\n---------------------------------\nHyperparameters\n\t" + '\n\t'.join(
            ["%s : %s" % (k, str(self.__dict__[k])) if not k.strip().startswith("rho") else "%s : %s" % (k, str(self.__dict__['rho'][k.split("rho")[1].strip()]))
             for k in self.hp_names]
        ) + "\n---------------------------------\n"


def _test_rho_matrix():
    '''
    Unit test for the Theta object, specific to the rho_matrix function.

    Checks for the following errors:

        1. Is the rho_matrix generated appropriately.
    '''
    # Get initial object setup
    theta = Theta()
    theta.rho = {"[0, 0]": 1, "[0, 1]": 0.5, "[1, 1]": 1}

    # Test this out, and compare with what it should be
    test = theta.rho_matrix([[0], [0], [0], [0], [1], [1], [0], [1]])
    ref = [
        [1.,  1.,  1.,  1.,  0.5, 0.5, 1.,  0.5],
        [1.,  1.,  1.,  1.,  0.5, 0.5, 1.,  0.5],
        [1.,  1.,  1.,  1.,  0.5, 0.5, 1.,  0.5],
        [1.,  1.,  1.,  1.,  0.5, 0.5, 1.,  0.5],
        [0.5, 0.5, 0.5, 0.5, 1.,  1.,  0.5, 1.],
        [0.5, 0.5, 0.5, 0.5, 1.,  1.,  0.5, 1.],
        [1.,  1.,  1.,  1.,  0.5, 0.5, 1.,  0.5],
        [0.5, 0.5, 0.5, 0.5, 1.,  1.,  0.5, 1.]
    ]

    return all([a == b for trow, rrow in zip(test, ref) for a, b in zip(trow, rrow)])


def run_unit_tests():
    assert _test_rho_matrix(), "pal.stats.hyperparameters.theta.rho_matrix() has failed."

