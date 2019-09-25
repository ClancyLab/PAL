import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cPickle as pickle

def get_plt(INFORMATION_SOURCE, ITER, RUN_INDEX, with_std=False):
    if not os.path.exists("imgs"):
        os.mkdir("imgs")

    # Get in the combo mapping
    combos = open("data_dumps/%d_combos_misokg.dat" % RUN_INDEX, 'r').read().strip().split('\n')
    combos = [c.split() for c in combos]
    combos = {'%s %s %s' % (k.split("Pb")[1].split("_")[0], k.split("Pb")[0], k.split("_")[-2]): int(index) - 240 * INFORMATION_SOURCE for index, k in combos if k.endswith(str(INFORMATION_SOURCE))}
    
    # Read in the dft data
    dft_data = pickle.load(open("enthalpy_N1_R3_Ukcal-mol", 'r'))
    
    # Read in the mean data we are predicting
    mu_values = open("data_dumps/%d_mu_misokg.dat_IS%d" % (RUN_INDEX, INFORMATION_SOURCE), 'r').read().strip().split('\n')[ITER]
    mu_values = [float(v) for v in mu_values.split()]
    if with_std:
        sig_values = open("data_dumps/%d_sig_misokg.dat_IS%d" % (RUN_INDEX, INFORMATION_SOURCE), 'r').read().strip().split('\n')[ITER]
        sig_values = [float(v)**0.5 for v in sig_values.split()]
    
    # Combine data
    if with_std:
        all_data = [(k, dft_data[k], mu_values[combos[k]], sig_values[combos[k]]) for k in combos.keys()]
    else:
        all_data = [(k, dft_data[k], mu_values[combos[k]]) for k in combos.keys()]
    all_data.sort(key=lambda x: x[1])
    
    # Plot
    y_dft = [v[1] for v in all_data]
    y_miso = [-1.0 * v[2] for v in all_data]
    xvals = range(len(y_dft))
    if with_std:
        sig_miso = [v[3] for v in all_data]

    plt.plot(xvals, y_dft, label="DFT")
    plt.plot(xvals, y_miso, label="misoKG (iter %d, IS %d)" % (ITER, INFORMATION_SOURCE))

    if with_std:
        plt.fill_between(xvals, [y - sig for y, sig in zip(y_miso, sig_miso)], [y + sig for y, sig in zip(y_miso, sig_miso)], alpha=0.5)

    plt.legend()
    plt.savefig("imgs/%d_%d_%.03d.png" % (RUN_INDEX, INFORMATION_SOURCE, ITER))

    plt.close()


for RUN_INDEX in [1]:
#for RUN_INDEX in range(100):
    IS = 0
    niters = len(open("data_dumps/%d_mu_misokg.dat_IS%d" % (RUN_INDEX, IS), 'r').read().strip().split('\n')) - 1
    for i in range(niters):
        get_plt(IS, i, RUN_INDEX, with_std=True)
    
    os.system("convert -delay 20 -loop 0 $(ls -v imgs/%d_%d_*.png) output_IS%d_RI%d.gif" % (RUN_INDEX, IS, IS, RUN_INDEX))

