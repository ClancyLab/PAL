# From python
import os
import numpy as np

# From Squid
from squid import geometry
from squid import structures

# From PAL
import pal
#import pal.stats
#import pal.utils
#import pal.kernels
import pal.constants.world as WORLD_CONSTS


def test_path_variables():
    PATHS = [
        WORLD_CONSTS.BINDING_ENERGY_DATABASE,
        WORLD_CONSTS.STRUCTURES_PATH
    ]
    return all([
        os.path.isdir(s)
        for s in PATHS if s is not None
    ] + [
        s.endswith("/")
        for s in PATHS if s is not None
    ])


def test_change_in_theory():
    return all([
        x == y for x, y in zip(
            ["! HF SV ECP{def2-TZVP}",
             "! OPT B97-D3 SV GCP(DFT/TZ) ECP{def2-TZVP} Grid7 SlowConv",
             "! OPT B97-D3 def2-TZVP GCP(DFT/TZ) ECP{def2-TZVP} Grid7 SlowConv",
             "! OPT PW6B95 def2-TZVP GCP(DFT/TZ) ECP{def2-TZVP} Grid7 SlowConv"],
            WORLD_CONSTS.default_routes)
    ] + ['''%geom
    MaxIter 1000
    end
''' == WORLD_CONSTS.extra_section])


def run_unit_tests():
    # The stats module
    pal.stats.MAP.run_unit_tests()
    pal.stats.MLE.run_unit_tests()
    pal.stats.priors.run_unit_tests()
    pal.stats.bayesian.run_unit_tests()
    pal.stats.likelihood.run_unit_tests()
    pal.stats.hyperparameters.run_unit_tests()
    pal.stats.knowledge_gradient.run_unit_tests()

    # The acquisition module
    pal.EI.run_unit_tests()
    pal.kg.run_unit_tests()
    pal.misokg.run_unit_tests()

    # The Kernels modules
    pal.kernels.matern.run_unit_tests()
    pal.kernels.periodic.run_unit_tests()

    # The Utils modules
    pal.utils.parser.run_unit_tests()
    pal.utils.strings.run_unit_tests()
    pal.utils.generate_systems.run_unit_tests()

    # Additional unit tests
    assert test_path_variables(), "pal.constants.world has invalid path variables."
    assert test_change_in_theory(), "A change in pal.constants.world.route or .extra_section means compatability issues with the database!"

    print("\nSUCCESS! ALL TESTS HAVE PASSED.")


if __name__ == "__main__":
    run_unit_tests()
