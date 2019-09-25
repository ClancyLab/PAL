import os
import numpy as np

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


for folder in ["data_dumps"]:
#for folder in ["MLE_data_dumps_1_to_0.1", "MAP_data_dumps_1_to_0.1", "MLE_data_dumps_1_to_0.01", "MAP_data_dumps_1_to_0.01", "MAP_data_dumps_1_to_0.1_fixed_rho", "MLE_data_dumps_1_to_0.1_fixed_rho"]:
    print("-----------------------------------------")
    print("For folder %s..." % folder)
    print("-----------------------------------------")

    costs = {"misokg": []}
    #costs = {"misokg": [], "kg": [], "ei": []}
    #if "0.01" in folder or "rho" in folder:
    #    costs = {"misokg": []}
    
    for fptr in os.listdir(folder):
        for sim in costs.keys():
            if fptr.endswith("_%s.log" % sim):
                cost = get_cost(folder + "/" + fptr)
                if cost is not None:
                    costs[sim].append(cost)
    for k in costs:
        avg = np.mean(costs[k])
        std = np.std(costs[k])
        nf = sorted(costs[k])[int(len(costs[k]) * 0.95)]
        ne = sorted(costs[k])[int(len(costs[k]) * 0.98)]
        print("%s -> %.3f +/- %.3f" % (k, avg, std))
        print("\t 95%% = %.3f, 98%% = %.3f" % (nf, ne))
    print("-----------------------------------------\n")

