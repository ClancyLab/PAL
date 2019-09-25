SAVEFIG = True
UPPER = 100

import os
import numpy as np
import cPickle as pickle
if SAVEFIG:
    import matplotlib
    matplotlib.use("Agg")
from matplotlib import pyplot as plt


def rosenbrock(x1, x2):
    return (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3

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
        X = map(lambda s: eval(s.split("],")[0] + "]")[1:], out)
        out = map(lambda s: s.split("Cost = ")[-1], data)
        cost = map(lambda s: float(s.split(",")[0]), out)
        return cost, X
    except (SyntaxError, IOError), e:
        print("Failed to parse %s/%d_%s.log" % (FOLDER, index, sffx))
        return None


def get_index_first_above_y(cost, X, y, default):
    for c, x in zip(cost, X):
        if rosenbrock(*x) < y:
            return float(c)
    return default


def get_index_first_above_c(cost, X, c):
    for cost, x in zip(cost, X):
        if cost >= c:
            return rosenbrock(*x)
    return rosenbrock(*X[-1])


def analysis_1():        
    NAMES = ["misokg", "bunit", "vl", "svl"]
    data = {name: [] for name in NAMES}
    y = -450.0
    default = 10000

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
    NAMES = ["misokg", "bunit", "vl", "svl"]
    ls.update({k: ls[aliases_labels[k]] for k in NAMES})
    lc.update({k: lc[aliases_labels[k]] for k in NAMES})
    y_values = np.arange(-456.3, -450.0, 0.1)
    y_values = y_values[::-1]
    default = 10000

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
    if SAVEFIG:
        plt.savefig("analysis_2.png")
    else:
        plt.show()


def analysis_3(rerun=False):
    NAMES = ["misokg", "bunit", "vl", "svl"]
    ls.update({k: ls[aliases_labels[k]] for k in NAMES})
    lc.update({k: lc[aliases_labels[k]] for k in NAMES})
    cost_values = np.arange(0, 10000, 1)

    if rerun and os.path.exists("analyzed_data_3.dat"):
        os.system("rm analyzed_data_3.dat")
    if rerun and os.path.exists("analyzed_data_se_3.dat"):
        os.system("rm analyzed_data_se_3.dat")

    if os.path.exists("analyzed_data_3.dat"):
        data = pickle.load(open("analyzed_data_3.dat", 'r'))
        data_se = pickle.load(open("analyzed_data_se_3.dat", 'r'))
    else:
        data = {name: [] for name in NAMES}
        data_se = {name: [] for name in NAMES}
        for sffx in NAMES:
            all_raw = [read_data(i, sffx) for i in range(UPPER)]
            for c in cost_values:
                tmp_data = []
                for i in range(UPPER):
                    raw = all_raw[i]
                    if raw is None:
                        continue
                    cost, X = raw
                    tmp_data.append(get_index_first_above_c(cost, X, c))
                m = np.mean(tmp_data)
                s = np.std(tmp_data)
                data[sffx].append(m)
                data_se[sffx].append(s / np.sqrt(float(UPPER)))
        pickle.dump(data, open("analyzed_data_3.dat", 'w'))
        pickle.dump(data_se, open("analyzed_data_se_3.dat", 'w'))

    for sffx in NAMES:
        data[sffx] = np.array(data[sffx])
        data_se[sffx] = np.array(data_se[sffx])

    for sffx in NAMES:
        plt.plot(cost_values, data[sffx], label=aliases_labels[sffx], linestyle=ls[sffx], color=lc[sffx], linewidth=3)
        plt.fill_between(cost_values, data[sffx] - data_se[sffx], data[sffx] + data_se[sffx], color=lc[sffx], alpha=0.5)
    plt.legend()
    plt.xlabel("Cost")
    plt.ylabel("Average Function Value first Exceeding Cost")
    if SAVEFIG:
        plt.savefig("analysis_3.png")
    else:
        plt.show()

#analysis_2(rerun=False)
analysis_3(rerun=False)
