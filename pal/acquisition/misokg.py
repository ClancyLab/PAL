'''
This file contains the Multi Information Source Optimization Knowledge Gradient
code, dubbed misoKG.
'''
import numpy as np

from pal.stats.knowledge_gradient import compute_kg_new


def _get_b(K, IS_LISTS, IS_l, x, err):
    '''
    Given a covariance matrix (K), a breakdown of which indices map to which
    information source (IS_LISTS), and which information source (IS_l) and
    point (xprime) we are sampling, return b for the kg factor calculation.

    b is calculated by first finding the vector K[(0, x'), (l, x)] where x'
    (written as xprime in variable form) is the point to be sampled, and l
    is the corresponding information source it is to be sampled from.

    Next, we scale this point-wise by the standard deviation + err of the
    information source l.  This is essentially:
        np.sqrt( np.diag(K[(l, x), (l, x)]) + noise )
    '''
    numer = K[np.ix_(IS_LISTS[0], IS_LISTS[IS_l])][:, x]
    denom = np.sqrt(K[np.ix_(IS_LISTS[IS_l], IS_LISTS[IS_l])][x, x] + err)

    return numer / denom


def getNextSample_misokg(mu, K, _1, _2, costs, X, samples, save=None, err=1E-6):
    '''
    MISOKG

    We compute the KG factor as in the MISO paper. Note that we wonder how observing x at IS s will affect the optimum at IS 0 (not at s!!!).
    The algorithm is easy: compute the KG factor for every composition x at IS 0 and let x_0 in argmax_x' KG(0,x').

    Then the same for IS 1: let x_1 in argmax_x' KG(1,x').let (s^{n+1},x^{n+1}) in max{ KG(0,x_0)/cost(IS0) , KG(1,x_1) / cost(IS1) },
    where cost is the comp. cost set by us for computing BE at this IS.

    When we look at Peter's KG 2009 paper, we see that it consists of 2 algorithms that are invoked for every KG(x) computation.
    '''

    IS_LISTS = [
        [
            i for i in range(len(X)) if X[i][0] == j
        ]
        for j in range(int(np.array(X)[:, 0].max()) + 1)
    ]
    IS0 = IS_LISTS[0]

    all_best = []
    for is_lvl, IS_x, c in zip(range(len(IS_LISTS)), IS_LISTS, costs):
        # For each x' in domain D, we find KG / c where KG = h(a, b) and c = cost
        #
        #     a = mu at IS0
        #     b = K[(0, x'), (l, x)] / (err + K[(l, x), (l, x)]) where x' is what we're looping over
        #         As such, the numerator is the vector of row x' in the sub-block K[(IS0), (ISl)]
        #         and the denominator is the diagonal of the K[(ISl), (ISl)] sub-block.
        #         We note that numpy does element-wise division here.
        #
        # By only doing compute_kg_new for non-sampled points, we can increase speed of calculation
        best = [
            (index_global,
             is_lvl,
             compute_kg_new(
                mu[IS0],
                _get_b(K, IS_LISTS, is_lvl, x, err)
             )[0] / c
            )
            for x, index_global in enumerate(IS_x)
            if index_global not in samples
        ]

        # In the case that we have fully sampled an IS, and best is empty, continue
        if len(best) == 0:
            continue

        if save is not None:
            fptr = open(save + "_IS" + str(is_lvl), 'a')
            fptr.write('\n' + ' '.join(["%f" % (float(b[-1]) * c) for b in best]))
            fptr.close()

        best = sorted(best, key=lambda v: v[2])
        all_best.append(best[-1])

    # If the hyperparameters are so bad that everything is nan, then randomly choose
    if all([np.isnan(v[2]) for v in all_best]):
        print("WARNING - Randomly choosing next sample point in misoKG.")
        all_best = np.random.choice(all_best)
        return all_best[0]

    all_best = sorted(all_best, key=lambda v: v[2])
    if all_best[-1][0] in samples:
        print("\nERROR - Sampled same point twice via misokg\n")
        print("ALL BEST = %s" % str(all_best))
        print("BEST = %s" % str(best))        

    return all_best[-1][0]


def _test_get_b():
    '''
    A unit test to ensure that b is calculated correctly.
    '''
    EPS = 1E-4
    K = np.block([
        [np.ones((2, 1)) * 0.8, np.ones((2, 1)) * 0.75, np.ones((2, 1)) * 0.5, np.ones((2, 1)) * 0.4],
        [np.ones((2, 1)) * 0.4, np.ones((2, 1)) * 0.5, np.ones((2, 1)) * 0.7, np.ones((2, 1)) * 0.6]
    ])
    IS_LISTS = [[0, 1], [2, 3]]
    err = 1E-6

    b1 = _get_b(K, IS_LISTS, 0, 0, err)
    b1_should_be = [0.8 / (0.8 + err)**0.5, 0.8 / (0.8 + err)**0.5]
    chk1 = all([abs(i - j) < EPS for i, j in zip(b1, b1_should_be)]) 

    b2 = _get_b(K, IS_LISTS, 1, 0, err)
    b2_should_be = [0.5 / (0.7 + err)**0.5, 0.5 / (0.7 + err)**0.5]
    chk2 = all([abs(i - j) < EPS for i, j in zip(b2, b2_should_be)]) 

    return chk1 and chk2


def run_unit_tests():
    assert _test_get_b(), "pal.acquisition.misokg.get_b() has failed."


if __name__ == '__main__':
    run_unit_tests()

