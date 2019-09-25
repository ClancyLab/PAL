SAVEFIG = False
UPPER = 2

import os
import numpy as np
import cPickle as pickle
if SAVEFIG:
    import matplotlib
    matplotlib.use("Agg")
from matplotlib import pyplot as plt


linestyles = [
    (0, ()),
    (0, (1, 1)),

    (0, (5, 10)),
    (0, (5, 5)),

    (0, (3, 1, 1, 1)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (5, 1)),

    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (1, 10)),
    (0, (1, 5)),
]

colors = [
    'green',
    'navy',
    'dodgerblue',
    'orange',
    'goldenrod',
    'black',
    'purple',
    'pink',
    'mediumblue',
    'lightgreen',
    'seagreen',
    'saddlebrown',
    'red',
]

ALL_NAMES = [
    "Bonilla - Dynamic Pearson",
    "Bonilla Miso PC Approach",
    "Bonilla - Very Loose",
    "Bonilla - Scaled, Very Loose",
    "Bonilla - Unit",
    "Bonilla - Identity",
    "EI",
    "misoKG",
#    "misoKG",
#    "Bonilla - Unit",
#    "Bonilla - Pearson",
#    "Bonilla - Scaled, Loose",
#    "Bonilla - Scaled, Very Loose",
#    "Bonilla - Very Loose",
#    "True Bonilla - Very Loose",
#    "Bonilla - Dynamic Pearson",
#    "EI",
#    "KG",
#    "Bonilla Miso Approach",
#    "Bonilla Miso PC Approach",
#    "Set HPs"
]
aliases_labels = {
    "bdpc": "Bonilla - Dynamic Pearson",
    "bidpc": "Bonilla Miso PC Approach",
    "bvl": "Bonilla - Very Loose",
    "bsvl": "Bonilla - Scaled, Very Loose",
    "bu": "Bonilla - Unit",
    "bI": "Bonilla - Identity",
    "ei": "EI",
    "misokg": "misoKG",
#    "b": "Bonilla - Very Loose",
#    "bl": "Bonilla - Scaled, Very Loose",
#    "tbvl": "True Bonilla - Very Loose",
#    "fbl": "Bonilla - Dynamic Pearson",
#    "bunit": "Bonilla - Unit",
#    "bpc": "Bonilla - Pearson",
#    "pc": "Bonilla - Pearson",
#    "kg": "KG",
#    "misokg": "misoKG",
#    "hope": "Bonilla Miso Approach",
#    "hopepc": "Bonilla Miso PC Approach",
#    "ideal": "Set HPs"
}
ls = {k: linestyles[i] for i, k in enumerate(ALL_NAMES)}
lc = {k: colors[i] for i, k in enumerate(ALL_NAMES)}

def read_data(index, sffx, FOLDER="data_dumps"):
    try:
        data = open("%s/%d_%s.log" % (FOLDER, index, sffx), 'r').read().strip().split("\n")
        out = map(lambda s: s.split("Recommendation = ")[-1], data)
        X = map(lambda s: eval(s.split("],")[0] + "]")[1:], out)
        out = map(lambda s: s.split("Cost = ")[-1], data)
        cost = map(lambda s: float(s.split(",")[0]), out)
        return cost, X
    except (SyntaxError, IOError), e:
        print("Failed to parse %s/%d_%s.log" % (FOLDER, index, sffx))
        return None

def get_index_first_above_y(cost, X, y, IS0):
    for c, x in zip(cost, X):
        if IS0(*x) < y:
            return float(c)
    return cost[-1]


def get_index_first_above_c(cost, X, c, IS0):
    for cost, x in zip(cost, X):
        if cost >= c:
            return IS0(*x)
    return IS0(*X[-1])


def analysis_1(folder, IS0, rerun=False):
    #NAMES = ["bdpc", "bidpc", "bvl", "bsvl", "bI", "ei"]
    NAMES = ["bdpc", "bidpc", "bvl", "bsvl", "bu", "bI", "ei", "misokg"]

    ls.update({k: ls[aliases_labels[k]] for k in NAMES})
    lc.update({k: lc[aliases_labels[k]] for k in NAMES})

    DELTA = 0.1
    y_values = np.arange(-470, -460, DELTA)
    y_values = y_values[::-1]

    if rerun and os.path.exists("%s_analyzed_data.dat" % folder):
        os.system("rm %s_analyzed_data.dat" % folder)
    if rerun and os.path.exists("%s_analyzed_data_se.dat" % folder):
        os.system("rm %s_analyzed_data_se.dat" % folder)

    if os.path.exists("%s_analyzed_data.dat" % folder):
        data = pickle.load(open("%s_analyzed_data.dat" % folder, 'r'))
        data_se = pickle.load(open("%s_analyzed_data_se.dat" % folder, 'r'))
    else:
        data = {name: [] for name in NAMES}
        data_se = {name: [] for name in NAMES}
        for sffx in NAMES:
            all_raw = [read_data(i, sffx, FOLDER=folder) for i in range(UPPER)]
            for y in y_values:
                tmp_data = []
                for i in range(UPPER):
                    raw = all_raw[i]
                    if raw is None:
                        continue
                    cost, X = raw
                    tmp_data.append(get_index_first_above_y(cost, X, y, IS0))
                m = np.mean(tmp_data)
                s = np.std(tmp_data)
 
                data[sffx].append(m)
                data_se[sffx].append(s / np.sqrt(float(UPPER)))
        pickle.dump(data, open("%s_analyzed_data.dat" % folder, 'w'))
        pickle.dump(data_se, open("%s_analyzed_data_se.dat" % folder, 'w'))

    for sffx in NAMES:
        data[sffx] = np.array(data[sffx])
        data_se[sffx] = np.array(data_se[sffx])

    for sffx in NAMES:
        plt.plot(y_values, data[sffx], label=aliases_labels[sffx], linestyle=ls[sffx], color=lc[sffx], linewidth=3)
        plt.fill_between(y_values, data[sffx] - data_se[sffx], data[sffx] + data_se[sffx], color=lc[sffx], alpha=0.5)
    plt.legend()
    plt.xlabel("Goal")
    plt.ylabel("Mean Cost")

    plt.gca().invert_xaxis()

    if SAVEFIG:
        plt.savefig("%s_analysis_1.png" % folder)
    else:
        plt.show()
    plt.close()

def analysis_2(folder, IS0, IS0_Cost, rerun=False):
    #NAMES = ["bdpc", "bidpc", "bvl", "bsvl", "bI", "ei"]
    NAMES = ["bdpc", "bidpc", "bvl", "bsvl", "bu", "bI", "ei", "misokg"]
    ls.update({k: ls[aliases_labels[k]] for k in NAMES})
    lc.update({k: lc[aliases_labels[k]] for k in NAMES})
    cost_values = np.arange(5 * IS0_Cost, 10100, 1)

    if rerun and os.path.exists("%s_analyzed_data_2.dat" % folder):
        os.system("rm %s_analyzed_data_2.dat" % folder)
    if rerun and os.path.exists("%s_analyzed_data_se_2.dat" % folder):
        os.system("rm %s_analyzed_data_se_2.dat" % folder)

    if os.path.exists("%s_analyzed_data_2.dat" % folder):
        data = pickle.load(open("%s_analyzed_data_2.dat" % folder, 'r'))
        data_se = pickle.load(open("%s_analyzed_data_se_2.dat" % folder, 'r'))
    else:
        data = {name: [] for name in NAMES}
        data_se = {name: [] for name in NAMES}
        for sffx in NAMES:
            all_raw = [read_data(i, sffx, FOLDER=folder) for i in range(UPPER)]
            for c in cost_values:
                tmp_data = []
                for i in range(UPPER):
                    raw = all_raw[i]
                    if raw is None:
                        continue
                    cost, X = raw
                    tmp_data.append(get_index_first_above_c(cost, X, c, IS0))
                m = np.mean(tmp_data)
                s = np.std(tmp_data)
                data[sffx].append(m)
                data_se[sffx].append(s / np.sqrt(float(UPPER)))
        pickle.dump(data, open("%s_analyzed_data_2.dat" % folder, 'w'))
        pickle.dump(data_se, open("%s_analyzed_data_se_2.dat" % folder, 'w'))

    for sffx in NAMES:
        data[sffx] = np.array(data[sffx])
        data_se[sffx] = np.array(data_se[sffx])

    for sffx in NAMES:
        plt.plot(cost_values, data[sffx], label=aliases_labels[sffx], linestyle=ls[sffx], color=lc[sffx], linewidth=3)
        plt.fill_between(cost_values, data[sffx] - data_se[sffx], data[sffx] + data_se[sffx], color=lc[sffx], alpha=0.5)
    plt.legend()
    plt.xlabel("Cost")
    plt.ylabel("Average Function Value first Exceeding Cost")
    plt.gca().invert_yaxis()
    if SAVEFIG:
        plt.savefig("%s_analysis_2.png" % folder)
    else:
        plt.show()
    plt.close()


RERUN = True

IS0 = lambda x1, x2: (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3
#IS0 = lambda x1, x2: -1.0 * rosenbrock(x1, x2)
COST = 1000.0

for nsample in [100, 200]:
    folder = "RNS%d" % nsample
#    analysis_1(folder, IS0, rerun=RERUN)
    analysis_2(folder, IS0, COST, rerun=RERUN)

