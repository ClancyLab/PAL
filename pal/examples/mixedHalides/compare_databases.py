import os
import cPickle as pickle
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

db1 = "enthalpy_N1_R3_Ukcal-mol"
db2 = "enthalpy_N1_R2_Ukcal-mol"

figname = "R2_vs_R3.png"

##############################################################################

# Error handle up front
assert os.path.exists(db1) and os.path.exists(db2), "Databases don't exist!"

# Read in the databases
data_1 = pickle.load(open(db1, 'r'))
data_2 = pickle.load(open(db2, 'r'))

# Parse into a single list of common ones, and sort
keys1 = sorted([k for k in data_1.keys() if k in data_2.keys()])
keys2 = sorted([k for k in data_2.keys() if k in data_1.keys()])
keys = keys1
if len(data_1) != len(data_2) or not all([k1 == k2 for k1, k2 in zip(keys1, keys2)]):
    print("Warning - Databases do not match 1-to-1. Will handle subset.")
    keys = sorted(list(set(keys1 + keys2)))
    data = [(data_1[k], data_2[k], k) for k in keys]
else:
    data = [(data_1[k], data_2[k], k) for k in keys]
data = sorted(data, key=lambda x: x[0])

# Find correlation coefficient
data1, data2, keys = zip(*data)
corr, p_value = pearsonr(data1, data2)
print("The correlation coefficient for these databases is %.2f with a pvalue of %.2f" % (corr, p_value))

# Plot
plt.plot(data1, label=db1)
plt.plot(data2, label=db2)
plt.ylabel("Energy (kcal/mol)")
plt.xlabel("Perovskite-Solvent Combination")
plt.legend()
#plt.show()
plt.savefig(figname)
