from pal.constants.solvents import solvents
from pal.constants.perovskites import perovskite_aliases


def simname_to_alias(name, warn_length=True, local_solvents=solvents):
    '''
    A function that shortens a name from the perovskite/solvent layout to an
    alias that is ideally under 32 characters long (as this is the limit for
    jobnames on the NBS queue).

    These are the potential names we need to shorten
        "MD_perov_solv_L": "P%s_S%s_l" % (perovskite, solvent),
        "MD_perov_solv_S": "P%s_S%s_%d_s" % (perovskite, solvent, num_solvents),
        "MD_solv_L": "S%s_l" % solvent,
        "MD_solv_S": "S%s_N%d_s" % (solvent, num_solvents),
        "DFT_perov": "P%s_R%d_I%d" % (perovskite, route_lvl, info_source),
        "DFT_perov_solv": "P%s_S%s_N%d_R%d_I%d" % (perovskite, solvent, num_solvents, route_lvl, info_source),
        "DFT_solv": "S%s_N%d_R%d_I%d" % (solvent, num_solvents, route_lvl, info_source)

    Identifiers are:
        P - Perovskite
        S - Solvent
        R - Route Level
        I - Information Source Level
        N - Number of Solvents

    **Parameters**

        name: *str*
            The name to be shortened.
        warn_length: *bool*
            A flag to turn on a warning that will raise an exception when the
            alias is longer than 31 characters.

    **Returns**

        alias: *str*
            A shorthand version of the name.
    '''
    split_name = name.strip().split("_")

    for i, part in enumerate(split_name):
        # Alias the perovskite name
        if part.startswith("P"):
            assert part[1:] in perovskite_aliases, "Perovskite name not found in perovskite_aliases."
            split_name[i] = "P%s" % str(perovskite_aliases[part[1:]])
        elif part.startswith("S"):
            if "-" in part:
                parts = part.split("-")
                for j, subpart in enumerate(parts):
                    end_index = [int(s.isdigit()) for s in subpart].index(1)
                    solv = subpart[int(j == 0):end_index]
                    parts[j] = "%sx%s" % (str(local_solvents[solv]['index']), str(subpart[end_index:]))
                split_name[i] = "S" + "-".join(parts)
            else:
                # If only one solvent, do this
                split_name[i] = "S%s" % str(local_solvents[part[1:]]['index'])
        elif part.startswith("R"):
            # Nothing to do here for now
            pass
        elif part.startswith("I"):
            # Nothing to do here for now
            pass
        elif part.startswith("N"):
            # Nothing to do here for now
            pass

    new_name = "_".join(split_name)

    if warn_length and len(new_name) >= 32:
        raise Exception("Name (%s) has been shorted (%s), but is still too long!" % (name, new_name))

    return new_name


def alias_to_simname(alias, local_solvents=solvents):
    '''
    A function that expands out a shorthand alias into a longer, more
    human readable and descriptive name.

    Identifiers are:
        P - Perovskite
        S - Solvent
        R - Route Level
        I - Information Source Level
        N - Number of Solvents

    **Parameters**

        alias: *str*
            The alias to be expanded.

    **Returns**

        name: *str*
            The full, human readable, name.
    '''
    split_name = alias.split("_")

    for i, part in enumerate(split_name):
        # Alias the perovskite name
        if part.startswith("P"):
            assert part[1:] in perovskite_aliases, "Perovskite name not found in perovskite_aliases."
            split_name[i] = "P%s" % str(perovskite_aliases[part[1:]])
        elif part.startswith("S"):
            if "-" in part:
                parts = part.split("-")
                for j, subpart in enumerate(parts):
                    end_index = subpart.index("x")
                    solv = subpart[int(j == 0):end_index]
                    parts[j] = "%s%s" % (str(local_solvents[solv]['name']), str(subpart[end_index + 1:]))
                split_name[i] = "S" + "-".join(parts)
            else:
                # If only one solvent, do this
                split_name[i] = "S%s" % str(local_solvents[part[1:]]['name'])
        elif part.startswith("R"):
            # Nothing to do here for now
            pass
        elif part.startswith("I"):
            # Nothing to do here for now
            pass
        elif part.startswith("N"):
            # Nothing to do here for now
            pass

    return "_".join(split_name)


def _test_simname_to_alias(local_solvents):
    names_and_aliases = [
        ("PMAPbClII_SDMSO0.5-ACE0.5_N20_s", "P29_S0x0.5-4x0.5_N20_s"),
        ("PMAPbClII_SDMSO0.5-MCR0.2-THTO0.3_N20_s", "P29_S0x0.5-5x0.2-6x0.3_N20_s"),
        ("PFAPbBrBrI_SDMSO0.5-ACE0.5_N20_s", "P13_S0x0.5-4x0.5_N20_s"),
        ("PFAPbBrBrI_SDMSO0.5-ACE0.5_N20_R2_I3_s", "P13_S0x0.5-4x0.5_N20_R2_I3_s")
    ]
    return all([simname_to_alias(name, local_solvents=local_solvents) == alias for name, alias in names_and_aliases])


def _test_alias_to_simname(local_solvents):
    names_and_aliases = [
        ("PMAPbClII_SDMSO0.5-ACE0.5_N20_s", "P29_S0x0.5-4x0.5_N20_s"),
        ("PMAPbClII_SDMSO0.5-MCR0.2-THTO0.3_N20_s", "P29_S0x0.5-5x0.2-6x0.3_N20_s"),
        ("PFAPbBrBrI_SDMSO0.5-ACE0.5_N20_s", "P13_S0x0.5-4x0.5_N20_s"),
        ("PFAPbBrBrI_SDMSO0.5-ACE0.5_N20_R2_I3_s", "P13_S0x0.5-4x0.5_N20_R2_I3_s")
    ]
    return all([alias_to_simname(alias, local_solvents=local_solvents) == name for name, alias in names_and_aliases])


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
    local_solvents.update({str(v['index']): v for _, v in local_solvents.items()})
    local_solvents.update({v['index']: v for _, v in local_solvents.items()})

    assert _test_simname_to_alias(local_solvents), "pal.utils.parser.simname_to_alias() failed."
    assert _test_alias_to_simname(local_solvents), "pal.utils.parser.alias_to_simname() failed."


if __name__ == "__main__":
    run_unit_tests()

