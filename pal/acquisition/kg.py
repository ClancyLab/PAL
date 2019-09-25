'''
This file contains the Knowledge Gradient code.
'''
import numpy as np

from pal.stats.knowledge_gradient import compute_kg_new


def getNextSample_kg(mu, K, _1, _2, _3, X, samples, save=None, err=1E-6):
    '''
    We find x that maximizes the KG factor.  We do so by computing the KG
    factor at all compositions x, and simply return the best.
    '''

    #all_best = [(i, compute_kg_new(mu, K[i, :] / np.sqrt(K[i, i] + err))[0]) for i, _ in enumerate(X)]

    # By only doing compute_kg_new for non-sampled points, we can increase speed of calculation
    all_best = [(i, compute_kg_new(mu, K[i, :] / np.sqrt(K[i, i] + err))[0]) for i, _ in enumerate(X) if i not in samples]

    if save is not None:
        fptr = open(save, 'a')
        fptr.write('\n' + ' '.join(["%f" % b[-1] for b in all_best]))
        fptr.close()


    if all([np.isnan(v[1]) for v in all_best]):
        print("WARNING - Randomly choosing next sample point in KG.")
        all_best = np.random.choice(all_best)
        return all_best[0]

    all_best = sorted(all_best, key=lambda v: v[1])[-1]
    return all_best[0]


def run_unit_tests():
    pass
