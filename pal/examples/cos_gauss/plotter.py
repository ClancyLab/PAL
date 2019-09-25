import numpy as np
import matplotlib

SAVE_FIG = False

if SAVE_FIG:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

rosenbrock = lambda x1, x2: (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3
FOLDER = "data_dumps"

def plot_gain(indices, sffx="misokg", best_found=False):
    '''
    A function to plot the gain of the rosenbrock function sampled
    '''

    yrange = [500, 0]
    furthest = 5001
    for i in indices:
        try:
            out = open("%s/%d_%s.log" % (FOLDER, i, sffx), 'r').read().strip().split("\n")
            out = map(lambda s: s.split("Recommendation = ")[-1], out)
            X = map(lambda s: eval(s.split("],")[0] + "]")[1:], out)
            out = open("%s/%d_%s.log" % (FOLDER, i, sffx), 'r').read().strip().split("\n")
            out = map(lambda s: s.split("Cost = ")[-1], out)
            cost = map(lambda s: float(s.split(",")[0]), out)
        except (SyntaxError, IOError), e:
            print("Failed to parse %s/%d_%s.log" % (FOLDER, i, sffx))
            continue
    
        X0 = X[0]
        #gain = [rosenbrock(*X0) - rosenbrock(*x) for x in X]
        gain = [- rosenbrock(*x) for x in X]
        if best_found:
            gain = [max(gain[:j]) if j > 0 else v for j, v in enumerate(gain)]

        plt.plot(cost, gain)

        furthest = max(cost + [furthest])
        yrange[0] = min(gain + [yrange[0]])
        yrange[1] = max(gain + [yrange[1]])
    plt.xlim(5000, furthest + 1)
    plt.ylim(yrange[0] - 0.5, yrange[1] + 0.5)
    plt.title(sffx)
    plt.xlabel("Total Cost (1000 for IS0, 1 for IS1)")
    if SAVE_FIG:
        plt.savefig(sffx)
    else:
        plt.show()
    plt.close()


def plot_val(indices, sffx="misokg", CUTOFF = 2.0):
    '''
    A function to plot the value of the recommendation.
    '''

    if not isinstance(indices, list):
        indices = [indices]

    all_x, all_y = [], []
    BEST = -1.0 * rosenbrock(1.0, 1.0)
    all_costs = []

    furthest = 5001
    for i in indices:
        try:
            out = open("%s/%d_%s.log" % (FOLDER, i, sffx), 'r').read().strip().split("\n")
            out = map(lambda s: s.split("Recommendation = ")[-1], out)
            X = map(lambda s: eval(s.split("],")[0] + "]")[1:], out)
            out = open("%s/%d_%s.log" % (FOLDER, i, sffx), 'r').read().strip().split("\n")
            out = map(lambda s: s.split("Cost = ")[-1], out)
            cost = map(lambda s: float(s.split(",")[0]), out)
        except SyntaxError:
            print("Failed to parse %s/%d_%s.log" % (FOLDER, i, sffx))
            continue
    
        val = [BEST - -1.0 * rosenbrock(*x) for x in X]
        # Now, we consider only the best found
        val = [min(val[:j]) if j > 0 else v for j, v in enumerate(val)]
        plt.plot(cost, val)

        try:
            all_costs.append(cost[[j for j, v in enumerate(val) if v < CUTOFF][0]])
        except:
            print val
            raise Exception("Could not find a value difference smaller than %f." % CUTOFF)

        furthest = max(cost + [furthest])

    all_costs.sort()
    print np.mean(all_costs), np.std(all_costs), all_costs[int(0.95 * len(all_costs))]
    print all_costs

    plt.xlim(5000, furthest + 1)
    plt.ylim(0, 10)
    plt.title(sffx)
    plt.show()


if __name__ == "__main__":
    val = False
    upper = 100
    cut = 500000

    sims = ["misokg", "smisokg", "smisokg2", "slmisokg2",
            "svlmisokg2"]
    if val:
        for sim in sims:
            plot_val(range(upper), sffx=sim, CUTOFF=cut)
    else:
        for sim in sims:
            plot_gain(range(upper), sffx=sim, best_found=False)

