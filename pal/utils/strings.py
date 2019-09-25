from squid import geometry
import warnings


HALIDE_STRS = ["Br", "Cl", "I"]
CATION_STRS = ["MA", "FA", "Cs"]


def parseName(name, name_has_IS=True):
    '''
    A function to split up a name, given as ABX1X2X3_S_IS into its components.

    **Parameters**

        name: *str*
            A given combination name.
        name_has_IS: *bool, optional*
            Whether the name has the IS level in it or not.

    **Returns**

        halides: *list, str*

        cation: *list, str*

        ion: *list, str*

        solvent: *str*

        info_level: *int or None*
            If the name has an IS level defined, it is also returned.
    '''
    ion = "Pb" if "pb" in name.lower() else "Sn"
    halides = name.split(ion)[-1].split("_")[0]
    halides = [
        ["Cl" for i in range(halides.count("Cl"))],
        ["Br" for i in range(halides.count("Br"))],
        ["I" for i in range(halides.count("I"))]
    ]
    halides = sorted([h for hh in halides for h in hh])
    cation = name.split(ion)[0]
    solvent = name.split("_")[-1 - int(name_has_IS)]
    info_level = None
    if name_has_IS:
        info_level = int(name.split("_")[-1])
    return halides, [cation], [ion], solvent, info_level


def parseNum(num, solvents, mixed_halides=False, num_has_IS=True, num_has_order_param=None, sort=True):
    '''
    Convert descriptor into a name.  Note, this is extensible to if the IS exists, or if it doesn't
    (determined by num_has_IS).  If the IS does exist, it MUST be the first index of num.

    **Parameters**

    **Returns**

        name: *str*
            String representation of this system.  This looks like:
                cation + "Pb" + H1 + H2 + H3 + "_" + solvent + "_" + IS
            If num_has_IS = False, then ("_" + IS) is left out
    '''

    if num_has_order_param is not None:
        raise Exception("parseNum deprecation.  Do not use num_has_order_param. Instead, pass without it.")

    offset = int(num_has_IS)

    h1 = HALIDE_STRS[num[offset:].index(1)]
    if not mixed_halides:
        h2, h3 = h1, h1
        offset += 3
    else:
        h2 = HALIDE_STRS[num[3 + offset:].index(1)]
        h3 = HALIDE_STRS[num[6 + offset:].index(1)]
        offset += 9

    c = CATION_STRS[num[offset:].index(1)]

    s = [v["name"] for k, v in solvents.items() if v["index"] == num[-1]][0]

    if sort:
        h1, h2, h3 = sorted([h1, h2, h3])

    if num_has_IS:
        return "%sPb%s%s%s_%s_%d" % (c, h1, h2, h3, s, num[0])
    else:
        return "%sPb%s%s%s_%s" % (c, h1, h2, h3, s)


def alphaToNum(data, solvents, mixed_halides=False, mixed_cations=False, N_CATIONS=(3, 1), N_IONS=(1, 1), no_mixed=None, name_has_IS=True):
    '''
    Given an input combination, return a numeric descriptor.

    **Parameters**

        data: *list, str*

        solvents: *dict, ...*

        N_HALIDES: *tuple, int, optional*
            The number of halides and halide descriptors we need.  As we
            commonly only care about Cl, Br, I, there are 3 halides.  Further,
            we only want to select 3, and thus there are 3 selections.  Thus,
            N_HALIDES = (n_choices, n_selections) = (3, 3).

        N_CATIONS: *tuple, int, optional*
            The number of cations and cation descriptors we need.  As we
            commonly only care about MA, FA, Cs, there are 3 cations.  Further,
            we only want to select 1, and thus there is 1 selection.  Thus,
            N_CATIONS = (n_choices, n_selections) = (3, 1).

        N_IONS: *tuple, int, optional*
            NOTE! CURRENTLY THIS IS NO LONGER USED UNTIL WE DECIDE TO FACTOR
            IN THE ALTERNATE FORM!

            The number of ions and ion descriptors we need.  As we
            commonly only care about Pb, there is 1 ion.  Further,
            we only want to select 1, and thus there is 1 selection.  Thus,
            N_IONS = (n_choices, n_selections) = (1, 1).

        no_mixed: *bool, optional*
            A boolean to deprecate the updates made to reduce this descriptor
            back into the pure halide form.

        name_has_IS: *bool, optional*
            Whether the name has the IS level in it or not.

    **Returns**

        descriptor: *tuple, int/float*
            Returns a tuple describing this system.  In order we have:
                Br, Cl, I, Br, Cl, I, Br, Cl, I, MA, FA, Cs, rho, eps, index
            Note, index is simply an id for which solvent is used.
            Note, if no_mixed is True, then:
                Br, Cl, I, MA, FA, Cs, rho, eps, index
    '''
    if no_mixed is not None:
        raise Exception("No_mixed is deprecated. Please use mixed_halides instead.")

    if mixed_cations:
        raise Exception("ERROR! We have statically coded in 3 for 3 cations in descriptor!")

    if isinstance(data, str):
        data = [data]

    descriptor = [
        [0 for i in range(
            int(name_has_IS) +  # For information level
            (9 if mixed_halides else 3) +
            3 +  # For cations
            3  # For solvent density, polarization, and index
        )] for _ in data]

    for index, name in enumerate(data):
        halide, cation, ion, solvent, is_lvl = parseName(name, name_has_IS=name_has_IS)

        if not mixed_halides:
            halide = [halide[0]]

        if name_has_IS:
            descriptor[index][0] = is_lvl

        for ii, h in enumerate(halide):
            descriptor[index][int(name_has_IS) + ii * 3 + HALIDE_STRS.index(h)] = 1

        for ii, c in enumerate(cation):
            descriptor[index][
                int(name_has_IS) +  # For information level
                (9 if mixed_halides else 3) +
                CATION_STRS.index(c)] = 1

        descriptor[index][-3:] = [solvents[solvent][s] for s in ["density",
                                                                 "dielectric",
                                                                 "index"]]

    return descriptor


def reduce_to_name(i, h, c, unique=True, sort=True):
    '''
    A function to consistently reduce a perovskite composition into a name.

    **Parameters**

        i: *str*
            The ion (Pb or Sn).
        h: *list, str or str*
            The halide combination in the perovskite. If a string is passed,
            all three are assumed to be the same
        c: *str*
            The cation in the perovskite.
        unique: *bool, optional*
            Whether to have unique names (CsPbClClBr) vs non-unique (CsPbBrCl2).

    **Returns**

        name: *str*
            The perovskite name
    '''

    if not isinstance(i, str) or not isinstance(c, str):
        raise Exception("Error - Reduce to name only works for single ion and cation systems.")

    if isinstance(h, list) and len(h) == 1:
        h = h[0]

    if sort and isinstance(h, list):
        h = sorted(h)

    if unique:
        if isinstance(h, str):
            h = h + h + h
        else:
            h = "".join(h)
        return c + i + h
    else:
        if isinstance(h, str):
            fname = c + i + h + "3"
        else:
            hh = [x + str(h.count(x)) if h.count(x) > 1
                  else x for x in geometry.reduce_list(h)]
            if sort:
                hh.sort()
            hh = "".join(hh)
            fname = c + i + hh
        return fname


def replace_all(S, a, b):
    '''
    A simple automated replace all for a string.
    '''
    while str(a) in S:
        S = S.replace(str(a), str(b))
    return S


def _test_reduce_to_name():
    return all([
        reduce_to_name(i, h, c, unique=True, sort=True) == name
        for i, h, c, name in [
            ("Pb", ["Cl"], "Cs", "CsPbClClCl"),
            ("Pb", "Cl", "Cs", "CsPbClClCl"),
            ("Pb", ["Cl", "Br", "I"], "MA", "MAPbBrClI"),
            ("Sn", ["Cl", "Cl", "Br"], "FA", "FASnBrClCl")
        ]
    ] + [
        reduce_to_name(i, h, c, unique=True, sort=False) == name
        for i, h, c, name in [
            ("Pb", ["Cl", "Br", "I"], "MA", "MAPbClBrI"),
            ("Sn", ["Cl", "Cl", "Br"], "FA", "FASnClClBr")
        ]
    ] + [
        reduce_to_name(i, h, c, unique=False, sort=True) == name
        for i, h, c, name in [
            ("Pb", ["Cl"], "Cs", "CsPbCl3"),
            ("Pb", "Cl", "Cs", "CsPbCl3"),
            ("Pb", ["Cl", "Br", "I"], "MA", "MAPbBrClI"),
            ("Sn", ["Cl", "Cl", "Br"], "FA", "FASnBrCl2")
        ]
    ] + [
        reduce_to_name(i, h, c, unique=False, sort=False) == name
        for i, h, c, name in [
            ("Pb", ["Cl"], "Cs", "CsPbCl3"),
            ("Pb", "Cl", "Cs", "CsPbCl3"),
            ("Pb", ["Cl", "Br", "I"], "MA", "MAPbClBrI"),
            ("Sn", ["Cl", "Cl", "Br"], "FA", "FASnCl2Br")
        ]
    ])


def _test_replace_all():
    return "Test THIS string THIS THIS THOSE THIS" == replace_all("Test THAT string THAT THIS THOSE THAT", "THAT", "THIS")


def _test_parseName():
    names = ["MAPbIII_ACE_1", "FAPbBrII_ACE_0", "CsPbBrClI_ACE_1"]
    results = [
        (['I', 'I', 'I'], ['MA'], ['Pb'], 'ACE', 1),
        (['Br', 'I', 'I'], ['FA'], ['Pb'], 'ACE', 0),
        (['Br', 'Cl', 'I'], ['Cs'], ['Pb'], 'ACE', 1)
    ]
    return all([parseName(name) == result for name, result in zip(names, results)])


def _test_parseNum(local_solvents):
    return all([
        parseNum(num, local_solvents, mixed_halides=True, sort=True) == name
        for num, name in zip([
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0.78, 20.7, 4],
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0.78, 20.7, 4],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1.2, 42.84, 6],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1.1, 40.24, 3],
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0.78, 20.7, 4]
        ], ["FAPbClClCl_ACE_0",
            "FAPbClClCl_ACE_1",
            "CsPbBrClI_THTO_0",
            "MAPbClII_GBL_0",
            "MAPbBrII_ACE_1"])
    ] + [
        parseNum(num, local_solvents, mixed_halides=False, sort=True)
        for num, name in zip([
            [1, 0, 0, 1, 0, 1, 0, 0.78, 20.7, 4],
            [1, 0, 1, 0, 0, 1, 0, 1.2, 42.84, 6],
            [0, 1, 0, 0, 1, 0, 0, 1.1, 40.24, 3]
        ], ["FAPbIII_ACE_1",
            "MAPbClClCl_THTO_1",
            "CsPbBrBrBr_GBL_0"])
    ])


def _test_alphaToNum(local_solvents):
    return all([
        x == y for x, y in zip(
            alphaToNum(
                ["FAPbClClCl_ACE_0",
                 "FAPbClClCl_ACE_1",
                 "CsPbBrClI_THTO_0",
                 "MAPbClII_GBL_0",
                 "MAPbBrII_ACE_1"],
                local_solvents, mixed_halides=True, name_has_IS=True),
            [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0.78, 20.7, 4],
             [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0.78, 20.7, 4],
             [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1.2, 42.84, 6],
             [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1.1, 40.24, 3],
             [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0.78, 20.7, 4]])
    ] + [
        x == y for x, y in zip(
            alphaToNum(
                ["FAPbClClCl_ACE_0",
                 "FAPbBrBrBr_ACE_1",
                 "CsPbIII_THTO_0",
                 "MAPbClClCl_GBL_0",
                 "MAPbBrBrBr_ACE_1"],
                local_solvents, mixed_halides=False, name_has_IS=True),
            [[0, 0, 1, 0, 0, 1, 0, 0.78, 20.7, 4],
             [1, 1, 0, 0, 0, 1, 0, 0.78, 20.7, 4],
             [0, 0, 0, 1, 0, 0, 1, 1.2, 42.84, 6],
             [0, 0, 1, 0, 1, 0, 0, 1.1, 40.24, 3],
             [1, 1, 0, 0, 1, 0, 0, 0.78, 20.7, 4]])
    ])
def run_unit_tests():
    # A factor to scale the density.  This is done as in reality the box isn't
    # as densly packed because of the solute in there.
    SCALE_DENSITY = 1.0
    
    # Solvents
    local_solvents = {
        "DMSO": {"name": "DMSO", "density": 1.0 * SCALE_DENSITY, "dielectric": 46.7, "index": 0},
        "DMF": {"name": "DMF", "density": 0.95 * SCALE_DENSITY, "dielectric": 36.7, "index": 1},
        "NMP": {"name": "NMP", "density": 1.1 * SCALE_DENSITY, "dielectric": 32.3, "index": 2},
        "GBL": {"name": "GBL", "density": 1.1 * SCALE_DENSITY, "dielectric": 40.24, "index": 3},
        "ACE": {"name": "ACE", "density": 0.78 * SCALE_DENSITY, "dielectric": 20.7, "index": 4},  # ACETONE
        "MCR": {"name": "MCR", "density": 0.85 * SCALE_DENSITY, "dielectric": 10.9, "index": 5},  # METHACROLEIN
        "THTO": {"name": "THTO", "density": 1.2 * SCALE_DENSITY, "dielectric": 42.84, "index": 6},
        "NM": {"name": "NM", "density": 1.14 * SCALE_DENSITY, "dielectric": 35.9, "index": 7},  # NITROMETHANE
    }
    assert _test_replace_all(), "pal.strings.replace_all() failed."
    assert _test_reduce_to_name(), "pal.strings.reduce_to_name() failed."
    assert _test_alphaToNum(local_solvents), "pal.strings.alphaToNum() failed."
    assert _test_parseName(), "pal.strings.parseName() failed"
    assert _test_parseNum(local_solvents), "pal.strings.parseNum() failed."


