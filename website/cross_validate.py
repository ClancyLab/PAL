import pickle
import numpy as np
from matplotlib import pyplot as plt


def name(k):
    '''
    Make sure we expect the data, if not, fix it
    '''
    h, c, s = k.strip().split()
    return "%sPb%s_%s" % (c, h, s)


def local_parse(lstring, offset):
    if len(lstring) == min([v for _, v in offset.items()]):
        held = {lstring: 1.0}
    else:
        i = 0
        held = {}
        while i < len(lstring):
            loffset = offset[lstring[i]]
            c = lstring[i: i + loffset]
            if not c in held:
                held[c] = 1.0
            i += loffset
            j = 0
            if len(lstring) > i + j:
                # Check if numeric
                if lstring[i + j].isdigit():
                    while i + j < len(lstring) and any([lstring[i + j].isdigit(), lstring[i + j] == "."]):
                        j += 1
                    held[c] = float(lstring[i: i + j])
            i += j

    # Normalize
    total = sum([v for _, v in held.items()])
    held = {k: v / total for k, v in held.items()}

    return held


def solv_parse(solv):
    held = {}
    local = ""
    val = ""
    flip = False
    for s in solv:
        if s.isalpha():
            if flip:
                held[local] = float(val)
                flip = False
                local = val = ""
            local += s
        else:
            flip = True
            val += s
    if val == "":
        if local != "":
            held[local] = 1.0
    elif local != "":
        held[local] = float(val)

    # Normalize
    total = sum([v for _, v in held.items()])
    held = {k: v / total for k, v in held.items()}

    return held


def split_name(perovskite):
    cation, other = perovskite.split("Pb")
    halide, solvent = other.split("_")

    # Handle breaking up the Cation even further
    cation = local_parse(cation, {"M": 2, "F": 2, "C": 2})
    halide = local_parse(halide, {"C": 2, "B": 2, "I": 1})
    solvent = solv_parse(solvent)

    return cation, halide, solvent


def get_energy_from_arithmetic_mean(perovskite):
    '''
    Given a perovskite name as MAPbIClBr_THTO (or something equiv), find the
    predicted intermolecular binding energy and return it.
    '''
    all_cations, all_halides, all_solvents = split_name(perovskite)

    total = 0.0
    for s_name, s_val in all_solvents.items():
        if not s_val:
            continue

        h_total = 0.0
        for h_name, h_val in all_halides.items():
            if not h_val:
                continue

            c_total = 0.0
            for c_name, c_val in all_cations.items():
                if not c_val:
                    continue

                name = "%sPb%s_%s" % (c_name, h_name * 3, s_name)
                c_total += c_val * data[name]

            h_total += h_val * c_total

        total += s_val * h_total

    return total


def get_energy_from_geometric_mean(perovskite):
    '''
    Given a perovskite name as MAPbIClBr_THTO (or something equiv), find the
    predicted intermolecular binding energy and return it.
    '''
    all_cations, all_halides, all_solvents = split_name(perovskite)

    total = 1.0
    for s_name, s_val in all_solvents.items():
        if not s_val:
            continue

        h_total = 1.0
        for h_name, h_val in all_halides.items():
            if not h_val:
                continue

            c_total = 1.0
            for c_name, c_val in all_cations.items():
                if not c_val:
                    continue

                name = "%sPb%s_%s" % (c_name, h_name * 3, s_name)
                c_total *= abs(data[name])**c_val

            h_total *= c_total**h_val

        total *= h_total**s_val

    return -1.0 * total


if __name__ == "__main__":
    # fname = "enthalpy_N1_R2_Ukcal-mol_v2"
    fname = "enthalpy_N1_R3_Ukcal-mol_v2"
    # fname = "enthalpy_N3_R2_Ukcal-mol_v2"
    # fname = "enthalpy_N5_R2_wo_GBL_Ukcal-mol_v2"
    data = pickle.load(open(fname, 'rb'))

    if not any(["Pb" in s for s in data.keys()]):
        data = {
            name(k): v for k, v in data.items()
        }

    d2 = {
        k: float("%.2f" % v) for k, v in data.items()
        if any([h in k for h in ["III", "ClClCl", "BrBrBr"]])
    }
    k, v = zip(*d2.items())

    # odd_solvents = ["CHCl3", "ETH", "PYR", "IPA",
    #                 "DMA", "FAM", "H2O", "CH3OH"]
    # odd_solvents = []
    # data = {
    #     k: v for k, v in data.items()
    #     if not any([s in k for s in odd_solvents])
    # }

    required_solv = ['DMF', 'NM', 'GBL', 'THTO', 'MCR', 'ACE', 'DMSO', 'NMP']
    # required_solv = ['THTO', 'MCR', 'ACE', 'DMF', 'NMP', 'NM']
    data = {
        k: v for k, v in data.items()
        if any([s in k for s in required_solv])
    }

    SOLVENTS = list(set([k.split("_")[-1] for k in data.keys()]))
    print("SOLVENTS:")
    print(SOLVENTS)
    print("Len dataset = %d" % len(data))

    # Predict everything
    all_data = [
        (name, data[name], get_energy_from_geometric_mean(name))
        # (name, data[name], get_energy_from_arithmetic_mean(name))
        for name in data.keys()
        if all(["III" not in name, "BrBrBr" not in name, "ClClCl" not in name])
    ]

    x_dft = np.array([d[1] for d in all_data])
    x_pred = np.array([d[2] for d in all_data])

    rms = np.sqrt(np.mean(np.square(x_dft - x_pred)))
    print("RMS = %.2f kcal/mol" % rms)

    plt.plot(x_dft, x_pred, 'r*')
    plt.xlabel("DFT Prediction (kcal/mol)")
    plt.ylabel("Arithmetic Mean Prediction (kcal/mol)")
    plt.show()
