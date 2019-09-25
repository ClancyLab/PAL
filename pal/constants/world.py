from squid import sysconst
from squid.structures import Struct
import numpy as np

# PAL Install Dir
PAL_DIR = "/home-2/hherbol1@jhu.edu/programs/PAL/"
#PAL_DIR = "/fs/home/hch54/PAL/"
#PAL_DIR = "/mnt/c/Users/hherbol/Documents/subsystem/PAL/"

# Random perterbation for likelihood calculation
RANDOM_PERTERBATION_LIKELIHOOD = 0
#RANDOM_PERTERBATION_LIKELIHOOD = 1E-5

# An identifier for a failed simulation
FAILED_SIMULATION = 123456789

# Path to Binding Energy database.  NOTE! IT MUST END IN A /
# If none exists, set to None
#BINDING_ENERGY_DATABASE = "/home-2/hherbol1@jhu.edu/my_scratch/BE_Database/database/"
BINDING_ENERGY_DATABASE = "/scratch/users/hherbol1@jhu.edu/MISO/energy_spread/local_db/"
#BINDING_ENERGY_DATABASE = PAL_DIR + "BE_Database/database/"
# BINDING_ENERGY_DATABASE = "/fs/home/hch54/Documents/Projects/PAL/BE_Database/database/"
DB_FPTR = PAL_DIR + "BE_Database/be_database.pickle"
# DB_FPTR = "/fs/home/hch54/Documents/Projects/PAL/BE_Database/be_database.pickle"

# Path to a folder containing any potential cml or xyz files.  If none exists,
# set to None.  NOTE! IT MUST END IN A /
#STRUCTURES_PATH = "/home-2/hherbol1@jhu.edu/my_scratch/BE_Database/structures/"
STRUCTURES_PATH = "/scratch/users/hherbol1@jhu.edu/MISO/energy_spread/structures/"
#STRUCTURES_PATH = PAL_DIR + "structures/"
#STRUCTURES_PATH = "/fs/home/hch54/Documents/Projects/PAL/structures/"

# Atom Types
atom_types = {
    "Pb": {'bond_count': 0, 'style': 'lj/cut', 'vdw_r': 3.81661, 'index': 356,
           'notes': 'Barium Ion Ba+2', 'element': 56, 'vdw_e': 0.047096,
           'charge': 2.0, 'mass': 137.33, 'index2': 76, 'element_name': 'Ba'},
    "Sn": {'bond_count': 0, 'style': 'lj/cut', 'vdw_r': 3.81661, 'index': 356,
           'notes': 'Barium Ion Ba+2', 'element': 56, 'vdw_e': 0.047096,
           'charge': 2.0, 'mass': 137.33, 'index2': 76, 'element_name': 'Ba'},
    "Cl": {'bond_count': 0, 'style': 'lj/cut', 'vdw_r': 4.02, 'index': 344,
           'notes': 'Chloride Ion Cl-', 'element': 17, 'vdw_e': 0.71,
           'charge': -1.0, 'mass': 35.453, 'index2': 21, 'element_name': 'Cl'},
    "Br": {'bond_count': 0, 'style': 'lj/cut', 'vdw_r': 4.28, 'index': 345,
           'notes': 'Bromide Ion Br-', 'element': 35, 'vdw_e': 0.71,
           'charge': -1.0, 'mass': 79.904, 'index2': 65, 'element_name': 'Br'},
    "I": {'bond_count': 0, 'style': 'lj/cut', 'vdw_r': 4.81, 'index': 346,
          'notes': 'Iodide Ion I-', 'element': 53, 'vdw_e': 0.71,
          'charge': -1.0, 'mass': 126.905, 'index2': 66, 'element_name': 'I'}
}

# WARNING! DO NOT CHANGE THE ORDER HERE!!! THIS WOULD BREAK THE NAMING CONVENTION
# USED WHEN DETERMING WHAT LEVEL OF THEORY A SIMULATION HAS
if sysconst.use_orca4:
    # Orca4
    default_routes = [
        "! HF def2-TZVP",
        "! OPT B97-D3 SV GCP(DFT/TZ) Grid7 SlowConv",
        "! OPT B97-D3 def2-TZVP GCP(DFT/TZ) Grid7 SlowConv",
        "! OPT PW6B95 def2-TZVP GCP(DFT/TZ) Grid7 SlowConv",
        "! OPT BLYP def2-TZVP GCP(DFT/TZ) Grid7 SlowConv",
        "! OPT BLYP NL def2-TZVP GCP(DFT/TZ) Grid7 SlowConv",
        "! OPT BLYP D3BJ def2-TZVP GCP(DFT/TZ) Grid7 SlowConv"
]
else:
    # Orca3
    default_routes = [
        "! HF SV ECP{def2-TZVP}",
        "! OPT B97-D3 SV GCP(DFT/TZ) ECP{def2-TZVP} Grid7 SlowConv",
        "! OPT B97-D3 def2-TZVP GCP(DFT/TZ) ECP{def2-TZVP} Grid7 SlowConv",
        "! OPT PW6B95 def2-TZVP GCP(DFT/TZ) ECP{def2-TZVP} Grid7 SlowConv"]

extra_section = '''%geom
    MaxIter 1000
    end
'''

# A function for getting a seed. Either static for repeatable simulations, or
# random for not
#get_seed = lambda: 12345
get_seed = lambda: int(10000 * np.random.random())

# Default angle types
default_angles = {"type": Struct(),
                  "angle": 110.7,
                  "style": "harmonic",
                  "e": 37.5,
                  "index2s": (13, 13, 46)}
