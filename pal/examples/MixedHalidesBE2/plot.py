import os
import csv
import numpy as np
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_local():
    combos = pickle.load(open("combos.dat", 'r'))

    os.system("cp mu_update.dat tmp.dat")
    os.system("cp mu_update_ei.dat tmp_ei.dat")

    os.system("mkdir -p imgs")
    os.system("rm imgs/*")

    data = open("tmp.dat", 'r').read().strip().split("\n")
    data_ei = open("tmp_ei.dat", 'r').read().strip().split("\n")
    for i, d in enumerate(data):
        if i == len(data_ei):
            break
        data[i] = [float(x) for x in d.strip().split()]
        data_ei[i] = [float(x) for x in data_ei[i].strip().split()]

        local = [(c[0], v, c[2]) for c, v in zip(combos, data[i])]
        k = 0
        for j, l in enumerate(local):
            if l[-1] == 0:
                local[j] = list(l) + [data_ei[i][k]]
                k += 1
        local = sorted(local, key=lambda x: x[0])

        IS0 = [l[0] for l in local if l[2] == 0]
        IS0_predict = [l[1] for l in local if l[2] == 0]
        IS1_predict = [l[1] for l in local if l[2] == 1]
        IS0_predict_EI = [l[3] for l in local if l[2] == 0]

        plt.plot(IS0, label="IS0")
        plt.plot(IS0_predict_EI, label="IS0 prediction (EI) at %d iter" % i)
        plt.plot(IS0_predict, label="IS0 prediction at %d iter" % i)
        plt.plot(IS1_predict, label="IS1 prediction at %d iter" % i)
        plt.legend()
        plt.savefig("imgs/%.04d.png" % i)
        plt.close()

    os.system("rm tmp.dat tmp_ei.dat")


def plot_queue():

    index = 0

    combos = pickle.load(open("combos.dat", 'r'))

    os.system("cp data_dumps/%d_mu.dat tmp.dat" % index)
    os.system("cp data_dumps/%d_mu_ei.dat tmp_ei.dat" % index)
    os.system("cp data_dumps/%d_mu_kg.dat tmp_kg.dat" % index)

    os.system("mkdir -p imgs")
    os.system("rm imgs/*")

    data = open("tmp.dat", 'r').read().strip().split("\n")
    data_ei = open("tmp_ei.dat", 'r').read().strip().split("\n")
    data_kg = open("tmp_kg.dat", 'r').read().strip().split("\n")
    for i, d in enumerate(data):
        if i == len(data_ei):
            break
        data[i] = [float(x) for x in d.strip().split()]
        data_ei[i] = [float(x) for x in data_ei[i].strip().split()]
        data_kg[i] = [float(x) for x in data_kg[i].strip().split()]

        local = [(c[0], v, c[2]) for c, v in zip(combos, data[i])]
        k = 0
        for j, l in enumerate(local):
            if l[-1] == 0:
                local[j] = list(l) + [data_ei[i][k]] + [data_kg[i][k]]
                k += 1
        local = sorted(local, key=lambda x: x[0])

        IS0 = [l[0] for l in local if l[2] == 0]
        IS0_predict = [l[1] for l in local if l[2] == 0]
        IS1_predict = [l[1] for l in local if l[2] == 1]
        IS0_predict_EI = [l[3] for l in local if l[2] == 0]
        IS0_predict_KG = [l[4] for l in local if l[2] == 0]

        plt.plot(IS0, label="IS0")
        plt.plot(IS0_predict_EI, label="IS0 prediction (EI) at %d iter" % i)
        plt.plot(IS0_predict_KG, label="IS0 prediction (KG) at %d iter" % i)
        plt.plot(IS0_predict, label="IS0 prediction at %d iter" % i)
        plt.plot(IS1_predict, label="IS1 prediction at %d iter" % i)
        plt.legend()
        plt.savefig("imgs/%.04d.png" % i)
        plt.close()

    os.system("rm tmp.dat tmp_ei.dat")


def plot_data_dumps(use_miso=True, use_ei=True, use_kg=True):
    '''
    This function will plot N plots, where N are all possible combinations of
    our system (the number of rows output in our dat files).  This specific
    plot will show the exact mean, the predicted EI, and predicted KG.
    '''

    # Ensure our image directory is empty for new images to be saved.
    os.system("mkdir -p imgs")
    os.system("rm imgs/*")

    # Read in the means and sigs of EI and KG
    lens = []
    if use_ei:
        ei_mu = np.loadtxt("data_dumps/0_mu_ei.dat", delimiter="\t")
        ei_sig = np.loadtxt("data_dumps/0_sig_ei.dat", delimiter="\t")
        lens.append(len(ei_mu))
    if use_miso:
        misokg_mu = np.loadtxt("data_dumps/0_mu_misokg.dat", delimiter="\t")
        misokg_sig = np.loadtxt("data_dumps/0_sig_misokg.dat", delimiter="\t")

        misokg_mu = [row[240:] for row in misokg_mu]
        misokg_mu = [row[240:] for row in misokg_sig]

        lens.append(len(misokg_mu))

    if use_kg:
        kg_mu = np.loadtxt("data_dumps/0_mu_kg.dat", delimiter="\t")
        kg_sig = np.loadtxt("data_dumps/0_sig_kg.dat", delimiter="\t")
        lens.append(len(kg_mu))

    cut = min(lens)

    print("\nCUTOFF = %d\n" % cut)
    if use_ei:
        ei_mu = ei_mu[:cut]
        ei_sig = ei_sig[:cut]
    if use_kg:
        kg_mu = kg_mu[:cut]
        kg_sig = kg_sig[:cut]
    if use_miso:
        misokg_mu = misokg_mu[:cut]
        misokg_sig = misokg_sig[:cut]

    # On the last iteration, we've sampled everything, so this is equivalent to
    # our dft results
    # mu = ei_mu[-1]
    # READ IN DFT
    mu = pickle.load(open("data_dumps/0_combos.dat", 'r'))
    mu = [-1.0 * v[0] for v in mu[::2]]

    # Loop through and plot
    for i, eim, eis, kgm, kgs, mkgm, mkgs in zip(range(len(ei_mu)), ei_mu, ei_sig, kg_mu, kg_sig, misokg_mu, misokg_sig):

        data = []
        data.append(mu)
        if use_ei:
            eim = ei_mu[i]
            eis = ei_sig[i]
            data.append(eim)
            data.append(eis)
        if use_kg:
            kgm = kg_mu[i]
            kgs = kg_sig[i]
            data.append(kgm)
            data.append(kgs)
        if use_miso:
            mkgm = misokg_mu[i]
            mkgs = misokg_sig[i]
            data.append(mkgm)
            data.append(mkgs)

        data = zip(*data)
        data = sorted(data, key=lambda x: x[0])

        if use_ei and use_kg and use_miso:
            val, eim, eis, kgm, kgs, mkgm, mkgs = zip(*data)
        elif use_ei and use_miso:
            val, eim, eis, mkgm, mkgs = zip(*data)
        elif use_ei and use_kg:
            val, eim, eis, kgm, kgs = zip(*data)
        elif use_ei and use_kg:
            val, eim, eis = zip(*data)
        elif use_kg and use_miso:
            val, kgm, kgs, mkgm, mkgs = zip(*data)
        elif use_kg:
            val, kgm, kgs= zip(*data)
        elif use_miso:
            val, mkgm, mkgs = zip(*data)

        plt.plot(val, label="DFT")

        if use_ei:
            plt.plot(eim, label="EI")
        if use_kg:
            plt.plot(kgm, label="KG")
        if use_miso:
            plt.plot(mkgm, label="misoKG")

        plt.legend()
        plt.xlabel("Combinations")
        plt.ylabel("Energy (kcal/mol)")
        plt.savefig("imgs/%.04d.png" % i)
        # plt.show()
        plt.close()

        print("Plotted %d." % i)


plot_data_dumps(use_kg=False, use_ei=False)
os.system('''ffmpeg -i imgs/%04d.png output.gif''')
