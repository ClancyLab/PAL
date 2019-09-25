'''
This file contains information on various solvents used, and their properties.
'''

# A factor to scale the density.  This is done as in reality the box isn't
# as densly packed because of the solute in there.
SCALE_DENSITY = 0.95

# Solvents
# https://depts.washington.edu/eooptic/linkfiles/dielectric_chart%5B1%5D.pdf
solvents = {
    "DMSO": {"name": "DMSO", "density": 1.0 * SCALE_DENSITY, "dielectric": 46.7, "index": 0, "polar_atoms": ["O"]},  # DIMETHYL SULFOXIDE
    "DMF": {"name": "DMF", "density": 0.95 * SCALE_DENSITY, "dielectric": 36.7, "index": 1, "polar_atoms": ["O"]},  # DIMETHYLFORMAMIDE
    "NMP": {"name": "NMP", "density": 1.1 * SCALE_DENSITY, "dielectric": 32.3, "index": 2, "polar_atoms": ["O"]},  # N-METHYL-2-PYRROLIDONE
    "GBL": {"name": "GBL", "density": 1.1 * SCALE_DENSITY, "dielectric": 40.24, "index": 3, "polar_atoms": ["O"]},  # GAMMA BUTYROLACTONE
    "ACE": {"name": "ACE", "density": 0.78 * SCALE_DENSITY, "dielectric": 20.7, "index": 4, "polar_atoms": ["O"]},  # ACETONE
    "MCR": {"name": "MCR", "density": 0.85 * SCALE_DENSITY, "dielectric": 10.9, "index": 5, "polar_atoms": ["O"]},  # METHACROLEIN
    "THTO": {"name": "THTO", "density": 1.2 * SCALE_DENSITY, "dielectric": 42.84, "index": 6, "polar_atoms": ["O"]},  # TETRAHYDROTHIOPHENE 1-OXIDE
    "NM": {"name": "NM", "density": 1.14 * SCALE_DENSITY, "dielectric": 35.9, "index": 7, "polar_atoms": ["O"]},  # NITROMETHANE
    "H2O": {"name": "H2O", "density": 1.0 * SCALE_DENSITY, "dielectric": 80.1, "index": 8, "polar_atoms": ["O"]},  # WATER
    "CH3OH": {"name": "CH3OH", "density": 0.79 * SCALE_DENSITY, "dielectric": 32.7, "index": 9, "polar_atoms": ["O"]},  # METHANOL
    "ETH": {"name": "ETH", "density": 0.79 * SCALE_DENSITY, "dielectric": 24.5, "index": 10, "polar_atoms": ["O"]},  # ETHANOL
    "CHCl3": {"name": "CHCl3", "density": 1.489 * SCALE_DENSITY, "dielectric": 4.81, "index": 11, "polar_atoms": ["Cl"]},  # CHLOROFORM
    "FAM": {"name": "FAM", "density": 1.133 * SCALE_DENSITY, "dielectric": 111.0, "index": 12, "polar_atoms": ["O", "N"]},  # FORMAMIDE
    "PYR": {"name": "PYR", "density": 0.983 * SCALE_DENSITY, "dielectric": 12.4, "index": 13, "polar_atoms": ["N"]},  # PYRIDINE
    "IPA": {"name": "IPA", "density": 0.786 * SCALE_DENSITY, "dielectric": 17.9, "index": 14, "polar_atoms": ["O"]},  # ISOPROPYL ALCOHOL
    "DMA": {"name": "DMA", "density": 0.937 * SCALE_DENSITY, "dielectric": 37.8, "index": 15, "polar_atoms": ["O"]},  # DIMETHYLACETAMIDE
}
solvents.update({str(v['index']): v for _, v in solvents.items()})
solvents.update({v['index']: v for _, v in solvents.items()})
