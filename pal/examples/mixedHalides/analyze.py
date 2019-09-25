SAVEFIG = True
UPPER = 100

import os
import numpy as np
import cPickle as pickle
if SAVEFIG:
    import matplotlib
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

IS0 = pickle.load(open("enthalpy_N1_R3_Ukcal-mol", 'r'))

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
    'seagreen',
    'purple',
    'pink',
    'mediumblue',
    'lightgreen',
    'saddlebrown',
    'red',
]

ALL_NAMES = [
    "misoKG",
    "Bonilla - Unit",
    "Bonilla - Pearson",
    "Bonilla - Scaled, Loose",
    "Bonilla - Scaled, Very Loose",
    "Bonilla - Very Loose",
    "EI",
    "KG"
]
aliases_labels = {
    "vl": "Bonilla - Very Loose",
    "bl": "Bonilla - Scaled, Very Loose",
    "svl": "Bonilla - Scaled, Very Loose",
    "bu": "Bonilla - Unit",
    "bunit": "Bonilla - Unit",
    "bpc": "Bonilla - Pearson",
    "pc": "Bonilla - Pearson",
    "ei": "EI",
    "kg": "KG",
    "misokg": "misoKG",
}
ls = {k: linestyles[i] for i, k in enumerate(ALL_NAMES)}
lc = {k: colors[i] for i, k in enumerate(ALL_NAMES)}

def read_data(index, sffx, FOLDER="data_dumps"):
    try:
        data = open("%s/%d_%s.log" % (FOLDER, index, sffx), 'r').read().strip().split("\n")
        out = map(lambda s: s.split("Recommendation = ")[-1], data)
        X = map(lambda s: s.split(",")[0], out)
        X = map(lambda s: s.split("Pb")[1].split("_")[0] + " " + s.split("Pb")[0] + " " + s.split("_")[-2], X)
        out = map(lambda s: s.split("Cost = ")[-1], data)
        cost = map(lambda s: float(s.split(",")[0]), out)
        return cost, X
    except (SyntaxError, IOError), e:
        print("Failed to parse %s/%d_%s.log" % (FOLDER, index, sffx))
        return None


def get_index_first_above_y(cost, X, y, default):
    for c, x in zip(cost, X):
        if IS0[x] < y:
            return float(c)
    return default


def analysis_1():
    NAMES = ["bpc", "bu", "bl", "ei", "kg"]
    #NAMES = ["bl", "bu", "bpc", "ei", "kg"]
    data = {name: [] for name in NAMES}
    y = -12.96
    default = len(IS0.keys()) * (1.0 + 0.1)  # Full cost

    for sffx in NAMES:
        for i in range(UPPER):
            raw = read_data(i, sffx)
            if raw is None:
                continue
            cost, X = raw
            data[sffx].append(get_index_first_above_y(cost, X, y, default))
        m = np.mean(data[sffx])
        s = np.std(data[sffx])
        i_95 = int(np.floor(0.95 * len(data[sffx])))
        p_95 = sorted(data[sffx])[i_95]
        print("%s: %.2f +/- %.2f, 95th is %.2f" %
              (sffx, m, s, p_95)
             )


def analysis_2(rerun=False):
    #NAMES = ["bpc", "bu", "bl", "ei", "kg"]
    NAMES = ["ei", "kg", "bu", "bpc", "bl"]

    ls.update({k: ls[aliases_labels[k]] for k in NAMES})
    lc.update({k: lc[aliases_labels[k]] for k in NAMES})

    y_values = np.arange(-13.0, -10.0, 0.1)
    y_values = y_values[::-1]
    default = len(IS0.keys()) * (1.0 + 0.1)  # Full cost

    if rerun and os.path.exists("analyzed_data.dat"):
        os.system("rm analyzed_data.dat")
    if rerun and os.path.exists("analyzed_data_se.dat"):
        os.system("rm analyzed_data_se.dat")

    if os.path.exists("analyzed_data.dat"):
        data = pickle.load(open("analyzed_data.dat", 'r'))
        data_se = pickle.load(open("analyzed_data_se.dat", 'r'))
    else:
        data = {name: [] for name in NAMES}
        data_se = {name: [] for name in NAMES}
        for sffx in NAMES:
            for y in y_values:
                tmp_data = []
                for i in range(UPPER):
                    raw = read_data(i, sffx)
                    if raw is None:
                        continue
                    cost, X = raw
                    tmp_data.append(get_index_first_above_y(cost, X, y, default))
                m = np.mean(tmp_data)
                s = np.std(tmp_data)
 
                data[sffx].append(m)
                data_se[sffx].append(s / np.sqrt(float(UPPER)))
        pickle.dump(data, open("analyzed_data.dat", 'w'))
        pickle.dump(data_se, open("analyzed_data_se.dat", 'w'))

    for sffx in NAMES:
        data[sffx] = np.array(data[sffx])
        data_se[sffx] = np.array(data_se[sffx])

    for sffx in NAMES:
        plt.plot(y_values, data[sffx], label=aliases_labels[sffx], linestyle=ls[sffx], color=lc[sffx], linewidth=3)
        plt.fill_between(y_values, data[sffx] - data_se[sffx], data[sffx] + data_se[sffx], color=lc[sffx], alpha=0.5)
    plt.legend()
    plt.xlabel("Goal")
    plt.ylabel("Mean Cost")

    plt.xlim(-12.1, -10.1)
    plt.ylim(10, 160)

    if SAVEFIG:
        plt.savefig("analysis_2.png")
    else:
        plt.show()

analysis_2(rerun=False)