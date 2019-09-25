from pal.opt import Optimizer
import pal.utils.strings as pal_strings
from pal.constants.solvents import solvents
from pal.kernels.matern import maternKernel52 as mk52
# from pal.objectives.binding_energy import get_binding_energy as BE
from pal.acquisition.misokg import getNextSample_misokg

import copy
# import random
import numpy as np
import cPickle as pickle

# Store data for debugging
IS0 = pickle.load(open("enthalpy_N3_R2_Ukcal-mol", 'r'))
IS1 = pickle.load(open("enthalpy_N1_R2_Ukcal-mol", 'r'))

# Generate the main object
sim = Optimizer()

# Assign simulation properties
###################################################################################################
# File names
sim.fname_out = "enthalpy.dat"
sim.fname_historical = None

# Information sources, in order from expensive to cheap
sim.IS = [
    lambda h, c, s: IS0[' '.join([''.join(h), c, s])],
    lambda h, c, s: IS1[' '.join([''.join(h), c, s])]
]
sim.costs = [
    2.0,
    1.0
]
sim.obj_vs_cost_fname = "obj_vs_cost_misokg.dat"
sim.save_extra_files = True
# sim.historical_nsample = 10
########################################
# Override the possible combinations with the reduced list of IS0
sim.combinations = [k[1] + "Pb" + k[0] + "_" + k[2] + "_" + str(IS) for k in [key.split() for key in IS0.keys()] for IS in range(len(sim.IS))]
combos_no_IS = [k[1] + "Pb" + k[0] + "_" + k[2] for k in [key.split() for key in IS0.keys()]]

# Because we do this, we should also generate our own historical sample
sim.historical_nsample = len(combos_no_IS)
choices = combos_no_IS
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

IS0 = np.array([x[-1] for x in data if x[0] == 0])
IS1 = np.array([x[-1] * 1.8 for x in data if x[0] == 1])

IS0, IS1 = zip(*sorted(zip(IS0, IS1)))

# IS0 = IS0 / np.linalg.norm(IS0)
# IS1 = IS1 / np.linalg.norm(IS1)

# print IS0
# print IS1

import matplotlib.pyplot as plt
plt.plot(IS0, label='IS0')
plt.plot(IS1, label='IS1')
plt.legend()
plt.show()

# print np.cov(IS0, IS1)[0][1]
print np.corrcoef(IS0, IS1)
