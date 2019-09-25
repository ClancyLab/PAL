import os
import numpy as np
import cPickle as pickle

def get_cost(fptr, goal="FAPbBrBrCl_THTO_0"):
    '''
    Read in a log file and try to find the cost it took to recommend something.
    '''
    goal = "Recommendation = %s" % goal

    buf = open(fptr, 'r').read().strip()
    if goal not in buf:
        return None

    cost = buf[buf.index(goal):].strip().split('\n')[0].strip()
    cost = cost[cost.index("Cost = ") + 6:].strip().split()[0].strip()
    if cost.endswith(","):
        cost = cost[:-1]

    return float(cost)


def get_cost_2(fptr, bench):
    buf = open(fptr, 'r').read().strip().split('\n')
    recs = [b.split(" Recommendation = ")[1].split(",")[0].strip() for b in buf if "Recommend" in b]
    cost = [float(b.split(" Cost = ")[1].split(",")[0].strip()) for b in buf if "Cost" in b]

    for i, c in enumerate(cost):
        if all([r in bench for r in recs[i:]]):
            return c
    return None


top_percent = 0.005
bench = pickle.load(open("enthalpy_N1_R3_Ukcal-mol", 'r'))
bench = [(v, k) for k, v in bench.items()]
bench = sorted(bench)
top_percent_index = int(top_percent * len(bench))
bench = bench[:top_percent_index]
bench = [b[1] for b in bench]
bench = [b.split()[1] + "Pb" + b.split()[0] + "_" + b.split()[-1] + "_0" for b in bench]

for folder in ["standard_ks", "scaled_ks", "forced_overlap_with_scaled_ks"]:
    print("-----------------------------------------")
    print("For folder %s..." % folder)
    print("-----------------------------------------")

    costs = {"misokg": [], "kg": [], "ei": []}

    for fptr in os.listdir(folder):
        for sim in costs.keys():
            if fptr.endswith("_%s.log" % sim):
                #cost = get_cost_2(folder + "/" + fptr, bench)
                cost = get_cost(folder + "/" + fptr)
                if cost is not None:
                    costs[sim].append(cost)

    for k in costs:
        if len(costs[k]) == 0:
            continue
        avg = np.mean(costs[k])
        std = np.std(costs[k])
        nf = sorted(costs[k])[int(len(costs[k]) * 0.95)]
        ne = sorted(costs[k])[int(len(costs[k]) * 0.98)]
        print("%s -> %.3f +/- %.3f" % (k, avg, std))
        print("    95%% = %.3f, 98%% = %.3f" % (nf, ne))
        print("    A total of %d data points were found." % len(costs[k]))
    print("-----------------------------------------\n")

