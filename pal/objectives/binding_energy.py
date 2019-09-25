# From Python
import os
import sys
import copy
import time
import types
import numpy as np

# From Squid
from squid import jobs
from squid import orca
from squid import units
from squid import files
from squid import sysconst
from squid import structures
from squid import lammps_job

# From PAL
import pal.log.database as logDB
import pal.utils.parser as parser
import pal.utils.strings as pal_strings
import pal.constants.world as WORLD_CONSTS
from pal.constants.solvents import solvents
from pal.utils.generate_systems import generate_ion_halide_cation as gen_perov
from pal.utils.generate_systems import strip_solvents


# Set our logger functions
# read_log, dump_log = logDB.read_from_db, logDB.dump_to_db
read_log, dump_log = logDB.read_pickle, logDB.dump_pickle
#BOX_SIZE = (100, 100, 100)
BOX_SIZE = (50, 50, 50)
SLEEPER = 60


def run_get_binding_energy(halide, cation, solvent, ion,
                           route_lvl, unit, num_solvents,
                           R_cutoff, rerun, run_sandboxed,
                           queue, nprocs, xhost, priority=None, verbose=False,
                           lammps_queue=None, lammps_nprocs=1,
                           override_seed=None, walltime_orca="3-00:00:00"):
    '''
    This function will actually run through the following procedure:

        1. Generate a list of all simulations necessary.
        2. Check for a database of simulations, and if the necessary ones
           already exist.  If so, then return or use what exists.
        3. Generate a large solvent box, with a solute in the middle, and
           equilibrate in MD.
        4. Strip solute and num_solvents (closest), and further equilibrate
           in MD.
        5. Optimize the geometries in DFT.
        6. Return difference in energies.

    **Parameters**

        halide: *list, str*

        cation: *str*

        solvent: *str* or *dict, str, float*
            The solvent of the system.  If using a mixed solvent, pass as a
            dictionary: {"ACETONE": 0.3, "GBL": 0.7}

        route_lvl: *int, optional*
            What level of theory (DFT) to run these calculations at.

        unit: *str, optional*
            What units the energy should be returned in.

        num_solvents: *int, optional*
            The number of solvents to use for enthalpy of solvation calculation.

        R_cutoff: *float, optional*
            The radial distance from the ion for which you want to select
            molecules for the calculation.

        rerun: *bool, optional*
            By default if an Orca simulation exists, it is used.  By setting
            rerun to True this will not be the case.

        run_sandboxed: *bool, optional*
            Whether to run this in a pseudo-sandbox.  This means that it will
            be run in such a way that this will not be added to the database,
            and no database will be checked for these values.

        queue: *str, optional*
            Which queue to run the simulation on.

        nprocs: *int, optional*
            How many processors to use.  Note, this is only in regards
            to the DFT jobs.

        override_seed: *int, optional*
            Whether to override the seed function manually with a given number.

        xhost: *list, str or str, optional*
            A list of processors, or a single processor for which to submit
            the simulations (on the queue).

        priority: *int, optional*
            A queue priority to ensure the subjobs run.  This may be necessary if
            many simulations are run in parallel and you are worried about locking
            the queue with too many parent jobs.

        verbose: *bool, optional*
            Whether to have a verbose output or not. If verbose, you'll know
            what calculation step is currently being run.

    **Return**

        BE: *float*
            The binding energy.
    '''

    # Error Handling
    if isinstance(solvent, dict):
        # Name convention - Longer than 32 char leads to failure to run on cluster!
        raise Exception("Error - Name convention issue with mixed solvents needs solving!")

    seed_func = WORLD_CONSTS.get_seed
    if override_seed is not None:
        seed_func = lambda: override_seed

    # Step 0 - Initialize values
    MD_RUN_LENGTH_L = 2000
    MD_RUN_LENGTH_S = 10000
    #POLAR_ATOM = "O"
    if isinstance(solvent, str):
        POLAR_ATOM = solvents[solvent]["polar_atoms"]
    else:
        POLAR_ATOM = []
        for solv in solvents.keys():
            POLAR_ATOM += solvents[solv]["polar_atoms"]
        POLAR_ATOM = list(set(POLAR_ATOM))

    energy_perov, energy_solv, energy_sys = None, None, None
    cml_perov, cml_solv = None, None
    md_perov_solv_L, md_perov_solv_S = None, None
    md_solv_L, md_solv_S = None, None

    route = WORLD_CONSTS.default_routes[route_lvl]
    extra_section = WORLD_CONSTS.extra_section

    # KEYWORDS: $RUN_NAME$, $ELEMENTS$, $MOBILE$, $IMOBILE$, $SEED$, $RUN_LEN$
    input_script = '''
    units real
    atom_style full
    pair_style lj/cut/coul/dsf 0.05 10.0 10.0
    bond_style harmonic
    angle_style harmonic
    dihedral_style opls

    boundary p p p
    read_data $RUN_NAME$.data

    dump 1 all xyz 100 $RUN_NAME$.xyz

    dump 2 all xyz 100 $RUN_NAME$_elems.xyz
    dump_modify 2 element $ELEMENTS$

    fix av all ave/time 1 1000 1000 c_thermo_pe
    thermo_style custom step f_av pe temp press
    thermo_modify flush yes
    thermo 1000

    group mobile id > $MOBILE$
    group mobile2 id > $MOBILE$
    group immobile subtract all mobile

    $IMOBILE$

    velocity mobile create 300.0 $SEED$ rot yes dist gaussian
    velocity immobile set 0.0 0.0 0.0

    $RELAX$

    write_restart $RUN_NAME$.restart
    '''

    relax_more = '''
    fix relax mobile nve/limit 0.001
    run 10000
    unfix relax

    fix relax mobile nve/limit 0.01
    run 10000
    unfix relax
    '''

    relax_normal = '''
    fix relax mobile nve/limit 0.1
    run 10000
    unfix relax

    timestep 1.0
    #fix motion mobile npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0 dilate mobile2
    fix motion mobile nvt temp 300.0 300.0 100.0
    run $RUN_LEN$
    unfix motion
    '''

    # Step 1 - Get a list of all simulations necessary.  Ex naming convention:
    #   MD Simulations
    #       CsPbBrClI_ACETONE_L      - A solvent box with perovskite, large
    #       CsPbBrClI_ACETONE_5_S    - A solvent box (num_solv) with perovskite, small
    #       ACETONE_L                - A solvent box, large
    #       ACETONE_5_S              - A solvent box, num_solv, small
    #   DFT Simulations
    #       CsPbBrClI_0              - Perovskite_Route
    #       CsPbBrClI_ACETONE_5_0    - Perovskite_Solvent_numSolvents_Route
    #       ACETONE_5_0              - Solvent_numSolvents_Route

    perovskite = pal_strings.reduce_to_name(ion, halide, cation, unique=True, sort=True)
    if verbose:
        print("Starting BE calculation for %s..." % perovskite)
        sys.stdout.flush()
    simulation_names = {
        "MD_perov_solv_L": "P%s_S%s_l" % (perovskite, solvent),
        "MD_perov_solv_S": "P%s_S%s_N%d_s" % (perovskite, solvent, num_solvents),
        "MD_solv_L": "S%s_l" % solvent,
        "MD_solv_S": "S%s_N%d_s" % (solvent, num_solvents),
        "DFT_perov": "P%s_R%d" % (perovskite, route_lvl),
        "DFT_perov_solv": "P%s_S%s_N%d_R%d" % (perovskite, solvent, num_solvents, route_lvl),
        "DFT_solv": "S%s_N%d_R%d" % (solvent, num_solvents, route_lvl)
    }

    # Convert names to aliases so we can avoid errors with long simulation names
    for k, v in simulation_names.items():
        simulation_names[k] = parser.simname_to_alias(v)

    # If we are running things "sandboxed" we append _SB_23421 where the
    # numerical suffix is random.
    if run_sandboxed:
        for k, v in simulation_names.items():
            simulation_names[k] = v + "_SB_" + str(seed_func())

    # Check if we have any of the above in the database, and act accordingly
    if WORLD_CONSTS.BINDING_ENERGY_DATABASE is not None:
        if verbose:
            print("\tAttempting to read in from database...")
            sys.stdout.flush()
        md_perov_solv_L = read_log(simulation_names["MD_perov_solv_L"])
        md_perov_solv_S = read_log(simulation_names["MD_perov_solv_S"])
        md_solv_L = read_log(simulation_names["MD_solv_L"])
        md_solv_S = read_log(simulation_names["MD_solv_S"])

        dft_perov_solv_data = read_log(simulation_names["DFT_perov_solv"])
        dft_perov_data = read_log(simulation_names["DFT_perov"])
        dft_solv_data = read_log(simulation_names["DFT_solv"])

        resub_dft_perov_solv = False
        resub_dft_perov = False
        resub_dft_solv = False

        if dft_perov_solv_data is not None and dft_perov_solv_data.converged:
            if verbose:
                print("\t\tFound DFT data (%s) for the Perovskite-Solvent system." % simulation_names["DFT_perov_solv"])
                sys.stdout.flush()
            energy_sys = dft_perov_solv_data.energy
        elif dft_perov_solv_data is not None:
            # We can resubmit from where it ended
            resub_dft_perov_solv = True

        if dft_perov_data is not None and dft_perov_data.converged:
            if verbose:
                print("\t\tFound DFT data (%s) for the perovskite system." % simulation_names["DFT_perov"])
                sys.stdout.flush()
            energy_perov = dft_perov_data.energy
        elif dft_perov_data is not None:
            # We can resubmit from where it ended
            resub_dft_perov = True

        if dft_solv_data is not None and dft_solv_data.converged:
            if verbose:
                print("\t\tFound DFT data (%s) for the solvent system." % simulation_names["DFT_solv"])
                sys.stdout.flush()
            energy_solv = dft_solv_data.energy
        elif dft_solv_data is not None:
            # We can resubmit from where it ended
            resub_dft_solv = True

    # If we have all the energies, return final value
    if all([E is not None for E in [energy_perov, energy_solv, energy_sys]]):
        if verbose:
            print("\tDONE! Final binding energy found to be %.4f Ha" % (energy_sys - energy_perov - energy_solv))
            sys.stdout.flush()
        return units.convert_energy("Ha", unit, energy_sys - energy_perov - energy_solv)

    # If not, make sure we have the base structures necessary for calculation

    # Read in the solvent structure if it is not already done
    if solvent + ".cml" not in [x for x in os.listdir(WORLD_CONSTS.STRUCTURES_PATH)]:
        raise Exception("Solvent (%s) CML file does not exist in STRUCTURES_PATH." % solvent)

    if verbose:
        print("\tGetting solvent molecule..."),
        sys.stdout.flush()
    PARAM_OBJ = None
    cml_solv = files.read_cml(
        WORLD_CONSTS.STRUCTURES_PATH + solvent + ".cml",
        new_method=False,
        test_charges=False, allow_errors=True,
        return_molecules=True, test_consistency=False,
        default_angles=WORLD_CONSTS.default_angles
    )[0]
    #cml_solv = cml_solv[0]
    if verbose:
        print("Done.")
        sys.stdout.flush()

    # Read in (or generate) perovskite structure
    if cml_perov is None:
        if verbose:
            print("\tGenerating perovskite system (%s)..." % simulation_names["DFT_perov"]),
            sys.stdout.flush()
        cml_perov = gen_perov(
            halide, cation,
            simulation_names["DFT_perov"],
            route, extra_section,
            ion="Pb", run_opt=True, priority=priority,
            walltime=walltime_orca, procs=nprocs
        )
        if verbose:
            print("Done.")
            sys.stdout.flush()

    if cml_solv is None:
        raise Exception("Unable to read in the CML file for the given solvent!")

    # Else, we need to submit simulations (in parallel on cluster) to get these values
    running_jobs = []

    if energy_sys is None:
        if resub_dft_perov_solv:
            print("\tResubmitting Perovskite-Solvent System from an unconverged one in the database.")
            md_perov_solv_S = structures.Molecule(dft_perov_solv_data.atoms)

        elif md_perov_solv_S is None:
            if md_perov_solv_L is None:
                if verbose:
                    print("\tRunning Large Perovskite-Solvent Lammps System..."),
                    sys.stdout.flush()

                # Generate a system object holding molecules for this simulation
                md_perov_solv_L = structures.System(box_size=BOX_SIZE, name=simulation_names["MD_perov_solv_L"])
                md_perov_solv_L.add(copy.deepcopy(cml_perov))
                md_perov_solv_L.packmol(
                    [cml_solv], molecule_ratio=(1,),
                    density=solvents[solvent]["density"],
                    seed=seed_func(),
                    new_method=False
                )
                # Assign types that otherwise don't exist
                for a in md_perov_solv_L.atoms:
                    if a.element in WORLD_CONSTS.atom_types and a.type is None:
                        a.type = structures.Struct()
                        a.type.__dict__.update(WORLD_CONSTS.atom_types[a.element])
                        a.type_index = a.type.index

                md_perov_solv_L.set_types(PARAM_OBJ)

                # Setup lammps script
                script = pal_strings.replace_all(input_script, "$RUN_NAME$", simulation_names["MD_perov_solv_L"])
                script = pal_strings.replace_all(script, "$ELEMENTS$", ' '.join([units.elem_i2s(a.element) for a in md_perov_solv_L.atom_types]))
                script = pal_strings.replace_all(script, "$IMOBILE$", "velocity immobile zero linear\nfix freeze immobile setforce 0.0 0.0 0.0")
                script = pal_strings.replace_all(script, "$RELAX$", relax_more + relax_normal)  # Moved from relax_normal to relax_more to stop exploding system
                script = pal_strings.replace_all(script, "$MOBILE$", len(cml_perov.atoms))
                script = pal_strings.replace_all(script, "$SEED$", seed_func())
                script = pal_strings.replace_all(script, "$RUN_LEN$", MD_RUN_LENGTH_L)

                # Run simulation
                J_md_perov_solv_L = lammps_job.job(
                    simulation_names["MD_perov_solv_L"],
                    script,
                    system=md_perov_solv_L,
                    redundancy=True,
                    queue=lammps_queue, procs=lammps_nprocs, no_echo=True)

                time.sleep(SLEEPER)
                J_md_perov_solv_L.wait()
                time.sleep(SLEEPER)

                # Post process and store data accordingly
                # 1. Read in equilibrated system to md_perov_solv_L
                # 2. LOG this system
                new_coordinates = files.read_xyz("lammps/%s/%s.xyz" % (simulation_names["MD_perov_solv_L"], simulation_names["MD_perov_solv_L"]))[-1]
                for a, b in zip(md_perov_solv_L.atoms, new_coordinates):
                    a.set_position(b.flatten())

                if not hasattr(J_md_perov_solv_L, "redundancy"):
                    dump_log(md_perov_solv_L, simulation_names["MD_perov_solv_L"])

                if verbose:
                    print("Done.")
                    sys.stdout.flush()

            if verbose:
                print("\tRunning Small Perovskite-Solvent Lammps System..."),
                sys.stdout.flush()
            # Take md_perov_solv_L and strip it to get md_perov_solv_S
            md_perov_solv_S = strip_solvents(md_perov_solv_L, R_cutoff, num_solvents, ion=ion, polar_atom=POLAR_ATOM)
            md_perov_solv_S.name = simulation_names["MD_perov_solv_S"]

            # Run MD for small system
            # Setup lammps script
            script = pal_strings.replace_all(input_script, "$RUN_NAME$", simulation_names["MD_perov_solv_S"])
            script = pal_strings.replace_all(script, "$ELEMENTS$", ' '.join([units.elem_i2s(a.element) for a in md_perov_solv_S.atom_types]))
            script = pal_strings.replace_all(script, "$IMOBILE$", "velocity immobile zero linear\nfix freeze immobile setforce 0.0 0.0 0.0")
            script = pal_strings.replace_all(script, "$RELAX$", relax_more)
            script = pal_strings.replace_all(script, "$MOBILE$", len(cml_perov.atoms))
            script = pal_strings.replace_all(script, "$SEED$", seed_func())
            script = pal_strings.replace_all(script, "$RUN_LEN$", MD_RUN_LENGTH_S)

            # Run simulation
            J_md_perov_solv_S = lammps_job.job(
                simulation_names["MD_perov_solv_S"],
                script,
                system=md_perov_solv_S,
                redundancy=True,
                queue=lammps_queue, procs=lammps_nprocs, no_echo=True)

            time.sleep(SLEEPER)
            J_md_perov_solv_S.wait()
            time.sleep(SLEEPER)

            # Post process and store data accordingly
            # 1. Read in equilibrated system to md_perov_solv_S
            # 2. LOG this system
            new_coordinates = files.read_xyz("lammps/%s/%s.xyz" % (simulation_names["MD_perov_solv_S"], simulation_names["MD_perov_solv_S"]))[-1]
            for a, b in zip(md_perov_solv_S.atoms, new_coordinates):
                a.set_position(b.flatten())
            if not hasattr(J_md_perov_solv_S, "redundancy"):
                dump_log(md_perov_solv_S, simulation_names["MD_perov_solv_S"])
            if verbose:
                print("Done.")
                sys.stdout.flush()

        if verbose:
            print("\tRunning Perovskite-Solvent System in DFT %s." % (["on the queue", "locally"][int(queue is None)]))
            sys.stdout.flush()
        # Take md_perov_solv_S and run it for orca
        J_dft_perov_solv = orca.job(
            simulation_names["DFT_perov_solv"],
            route, atoms=md_perov_solv_S.atoms,
            extra_section=extra_section, redundancy=True, priority=priority,
            walltime=walltime_orca,
            queue=queue, ntasks=nprocs, xhost=xhost)

        running_jobs.append((J_dft_perov_solv, "DFT_perov_solv"))

    if energy_solv is None:
        if resub_dft_solv:
            print("\tResubmitting Solvent System from an unconverged one in the database.")
            md_solv_S = structures.Molecule(dft_solv_data.atoms)
        elif md_solv_S is None:
            if md_solv_L is None:
                if verbose:
                    print("\tRunning Large Solvent Lammps System..."),
                    sys.stdout.flush()
                # Generate a system object holding molecules for this simulation
                md_solv_L = structures.System(box_size=BOX_SIZE, name=simulation_names["MD_solv_L"])
                md_solv_L.packmol(
                    [cml_solv], molecule_ratio=(1,),
                    density=solvents[solvent]["density"],
                    seed=seed_func(),
                    new_method=False
                )
                # Assign types that otherwise don't exist
                for a in md_solv_L.atoms:
                    if a.element in WORLD_CONSTS.atom_types and a.type is None:
                        a.type = structures.Struct()
                        a.type.__dict__.update(WORLD_CONSTS.atom_types[a.element])
                        a.type_index = a.type.index

                md_solv_L.set_types(PARAM_OBJ)

                # Setup lammps script
                script = pal_strings.replace_all(input_script, "$RUN_NAME$", simulation_names["MD_solv_L"])
                script = pal_strings.replace_all(script, "$ELEMENTS$", ' '.join([units.elem_i2s(a.element) for a in md_solv_L.atom_types]))
                script = pal_strings.replace_all(script, "$IMOBILE$", "")
                script = pal_strings.replace_all(script, "$RELAX$", relax_more + relax_normal)  # Moved from relax_normal to relax_more to stop exploding system
                script = pal_strings.replace_all(script, "$MOBILE$", 0)
                script = pal_strings.replace_all(script, "$SEED$", seed_func())
                script = pal_strings.replace_all(script, "$RUN_LEN$", MD_RUN_LENGTH_L)

                # Run simulation
                J_md_solv_L = lammps_job.job(
                    simulation_names["MD_solv_L"],
                    script,
                    system=md_solv_L, redundancy=True,
                    queue=lammps_queue, procs=lammps_nprocs, no_echo=True)

                time.sleep(SLEEPER)
                J_md_solv_L.wait()
                time.sleep(SLEEPER)

                # Post process and store data accordingly
                # 1. Read in equilibrated system to md_solv_L
                # 2. LOG this system
                new_coordinates = files.read_xyz("lammps/%s/%s.xyz" % (simulation_names["MD_solv_L"], simulation_names["MD_solv_L"]))[-1]
                for a, b in zip(md_solv_L.atoms, new_coordinates):
                    a.set_position(b.flatten())

                if not hasattr(J_md_solv_L, "redundancy"):
                    dump_log(md_solv_L, simulation_names["MD_solv_L"])

                if verbose:
                    print("Done.")
                    sys.stdout.flush()

            if verbose:
                print("\tRunning Small Solvent Lammps System..."),
                sys.stdout.flush()
            # Take md_solv_L and strip it to get md_solv_S
            md_solv_S = strip_solvents(md_solv_L, R_cutoff, num_solvents, ion=ion, polar_atom=POLAR_ATOM, no_solute=True)
            md_solv_S.name = simulation_names["MD_solv_S"]

            # Run MD for small system
            # Setup lammps script
            script = pal_strings.replace_all(input_script, "$RUN_NAME$", simulation_names["MD_solv_S"])
            script = pal_strings.replace_all(script, "$ELEMENTS$", ' '.join([units.elem_i2s(a.element) for a in md_solv_S.atom_types]))
            script = pal_strings.replace_all(script, "$IMOBILE$", "")
            script = pal_strings.replace_all(script, "$RELAX$", relax_more)
            script = pal_strings.replace_all(script, "$MOBILE$", 0)
            script = pal_strings.replace_all(script, "$SEED$", seed_func())
            script = pal_strings.replace_all(script, "$RUN_LEN$", MD_RUN_LENGTH_S)

            # Run simulation
            J_md_solv_S = lammps_job.job(
                simulation_names["MD_solv_S"],
                script,
                system=md_solv_S, redundancy=True,
                queue=lammps_queue, procs=lammps_nprocs, no_echo=True)

            time.sleep(SLEEPER)
            J_md_solv_S.wait()
            time.sleep(SLEEPER)

            # Post process and store data accordingly
            # 1. Read in equilibrated system to md_solv_S
            # 2. LOG this system
            new_coordinates = files.read_xyz("lammps/%s/%s.xyz" % (simulation_names["MD_solv_S"], simulation_names["MD_solv_S"]))[-1]
            for a, b in zip(md_solv_S.atoms, new_coordinates):
                a.set_position(b.flatten())

            if not hasattr(J_md_solv_S, "redundancy"):
                dump_log(md_solv_S, simulation_names["MD_solv_S"])

            if verbose:
                print("Done.")
                sys.stdout.flush()

        if verbose:
            print("\tRunning Solvent System in DFT %s." % (["on the queue", "locally"][int(queue is None)]))
            sys.stdout.flush()
        # Take md_solv_S and run it for orca
        J_dft_solv = orca.job(
            simulation_names["DFT_solv"],
            route, atoms=md_solv_S.atoms,
            extra_section=extra_section, redundancy=True, priority=priority,
            walltime=walltime_orca,
            queue=queue, ntasks=nprocs, xhost=xhost)

        running_jobs.append((J_dft_solv, "DFT_solv"))

    if energy_perov is None:
        if verbose:
            print("\tRunning Perovskite System in DFT %s." % (["on the queue", "locally"][int(queue is None)]))
            sys.stdout.flush()
        if resub_dft_perov:
            print("\tResubmitting Perovskite System from an unconverged one in the database.")
            cml_perov = structures.Molecule(dft_perov_data.atoms)
        # Run orca for perovskite
        J_dft_perov = orca.job(
            simulation_names["DFT_perov"],
            route, atoms=cml_perov.atoms,
            extra_section=extra_section, redundancy=True, priority=priority,
            walltime=walltime_orca,
            queue=queue, ntasks=nprocs, xhost=xhost)

        running_jobs.append((J_dft_perov, "DFT_perov"))

    bad_sim = False
    if verbose:
        print("\tWaiting on background dft simulations.")
        sys.stdout.flush()

    # A sleep timer is added prior to jobs being checked, so we ensure sacct
    # is up-to-date when running on systems like MARCC
    time.sleep(SLEEPER)
    for job, name in running_jobs:
        job.wait()
        time.sleep(SLEEPER)
        if verbose:
            print("\t\t%s has finished..." % name)
            sys.stdout.flush()
        # Post process, taking final structure and energy and logging it!
        data = orca.read(simulation_names[name])
        if not hasattr(job, "redundancy"):
            dump_log(data, simulation_names[name])

        if not data.converged:
            bad_sim = True

        if name == "DFT_perov_solv":
            energy_sys = data.energy
        elif name == "DFT_perov":
            energy_perov = data.energy
        elif name == "DFT_solv":
            energy_solv = data.energy
        else:
            raise Exception("Error should never be called.  What energy is this?")

    # In the case of a failed convergence, we return a specific value instead.
    if bad_sim:
        return WORLD_CONSTS.FAILED_SIMULATION

    if verbose:
        print("\tDONE! Final binding energy found to be %.4f Ha" % (energy_sys - energy_perov - energy_solv))
        sys.stdout.flush()
    # Now that we have all the energies, return final value
    return units.convert_energy("Ha", unit, energy_sys - energy_perov - energy_solv)


def get_binding_energy(halide, cation, solvent,
                       ion="Pb",
                       route_lvl=1,
                       unit="kT_300",
                       num_solvents=25,
                       R_cutoff=2000.0,
                       on_queue=None, rerun=False, run_sandboxed=False,
                       walltime_orca="3-00:00:00", walltime_pysub="3-00:00:00",
                       queue=sysconst.default_queue, nprocs=1, xhost=None,
                       lammps_queue=None, lammps_nprocs=1,
                       override_seed=None, priority=None, verbose=False):
    """
    This automates the process of calculating the binding energy between
    num_solvents and a given solute.

    **Parameters**

        halide: *list, str*

        cation: *str*

        solvent: *str* or *dict, str, float*
            The solvent of the system.  If using a mixed solvent, pass as a
            dictionary: {"ACETONE": 0.3, "GBL": 0.7}

        route_lvl: *int, optional*
            What level of theory (DFT) to run these calculations at.

        unit: *str, optional*
            What units the energy should be returned in.

        num_solvents: *int, optional*
            The number of solvents to use for enthalpy of solvation calculation.

        R_cutoff: *float, optional*
            The radial distance from the ion for which you want to select
            molecules for the calculation.

        on_queue: *str, optional*
            Whether to run this simulation on the queue.  If so, then specify
            the queue.  Else, None.

        rerun: *bool, optional*
            By default if an Orca simulation exists, it is used.  By setting
            rerun to True this will not be the case.

        run_sandboxed: *bool, optional*
            Whether to run this in a pseudo-sandbox.  This means that it will
            be run in such a way that this will not be added to the database,
            and no database will be checked for these values.

        queue: *str, optional*
            Which queue to run the simulation on.

        override_seed: *int, optional*
            Whether to override the seed function manually with a given number.

        nprocs: *int, optional*
            How many processors to use.  Note, this is only in regards
            to the DFT jobs.

        xhost: *list, str or str, optional*
            A list of processors, or a single processor for which to submit
            the simulations (on the queue).

        priority: *int, optional*
            A queue priority to ensure the subjobs run.  This may be necessary if
            many simulations are run in parallel and you are worried about locking
            the queue with too many parent jobs.

        verbose: *bool, optional*
            Whether to have a verbose output or not. If verbose, you'll know
            what calculation step is currently being run.

    **Return**

        If on_queue:

            job: *Job*
                A job object, pointing to the simulation in question.

        Else:

            BE: *float*
                The binding energy.
    """

    # If on_queue, then submit a python scrip to the cluster with the
    # simulation.  Else, just run it locally

    if run_sandboxed and override_seed is None:
        override_seed = int(100000 * np.random.random())

    if on_queue is not None:
        # Python script to submit to the cluster for this simulation
        pysub_str = """from pal.objectives.binding_energy import get_binding_energy
hsolv = get_binding_energy($HALIDE, "$CATION", "$SOLVENT",
            ion="$ION",
            route_lvl = $ROUTE_LVL,
            unit="$UNIT",
            num_solvents=$N_SOLVENTS,
            R_cutoff=$R_CUTOFF,
            rerun=$RERUN$, run_sandboxed=$RUN_SANDBOXED$,
            queue="$QUEUE", nprocs=$NPROCS, xhost=$XHOST,
            walltime_orca="$WALLTIME_ORCA", override_seed=$OVERRIDE_SEED,
            lammps_queue=$LAMMPS_QUEUE$, lammps_nprocs=$LAMMPS_NPROCS$,
            priority=$PRIORITY, verbose=$VERBOSE)"""

        if isinstance(halide, str):
            halide = [halide, halide, halide]
        pysub_str = pysub_str.replace("$HALIDE", str(halide))
        pysub_str = pysub_str.replace("$CATION", str(cation))
        pysub_str = pysub_str.replace("$SOLVENT", str(solvent))
        pysub_str = pysub_str.replace("$ION", str(ion))
        pysub_str = pysub_str.replace("$ROUTE_LVL", str(route_lvl))
        pysub_str = pysub_str.replace("$UNIT", str(unit))
        pysub_str = pysub_str.replace("$N_SOLVENTS", str(num_solvents))
        pysub_str = pysub_str.replace("$R_CUTOFF", str(R_cutoff))
        pysub_str = pysub_str.replace("$WALLTIME_ORCA", str(walltime_orca))
        pysub_str = pysub_str.replace("$RERUN$", str(rerun))
        pysub_str = pysub_str.replace("$RUN_SANDBOXED$", str(run_sandboxed))
        pysub_str = pysub_str.replace("$OVERRIDE_SEED", str(override_seed))

        pysub_str = pysub_str.replace("$LAMMPS_NPROCS$", str(lammps_nprocs))
        if lammps_queue is None:
            pysub_str = pysub_str.replace("$LAMMPS_QUEUE$", "None")
        else:
            pysub_str = pysub_str.replace("$LAMMPS_QUEUE$", '"%s"' % lammps_queue)

        pysub_str = pysub_str.replace("$QUEUE", str(queue))
        pysub_str = pysub_str.replace("$NPROCS", str(nprocs))
        pysub_str = pysub_str.replace("$PRIORITY", str(priority))
        pysub_str = pysub_str.replace("$VERBOSE", str(verbose))
        if isinstance(xhost, str):
            pysub_str = pysub_str.replace("$XHOST", '"%s"' % xhost)
        else:
            pysub_str = pysub_str.replace("$XHOST", str(xhost))

        # Get the name for this simulation
        job_name = "J_" + pal_strings.reduce_to_name(ion, halide, cation) + "_" + solvent + "_" + str(route_lvl)
        if run_sandboxed:
            job_name += "_SB_" + str(override_seed)
        job_name += ".py"

        # Write the file
        fptr = open(job_name, "w")
        fptr.write(pysub_str)
        fptr.close()

        # Run the simulation on the cluster
        running_job = jobs.pysub(job_name,
                                 nprocs=lammps_nprocs,
                                 queue=queue,
                                 walltime=walltime_pysub,
                                 xhost=xhost,
                                 path=os.getcwd(),
                                 modules=["pal"],
                                 remove_sub_script=True)

        # Here we define a function to attach to the job class.  This allows
        # the user to call "x.mbo()" to retrieve the mbo when the job is done.
        def result(self, unit="kT_300"):
            if not self.is_finished():
                return None
            else:
                perovskite = pal_strings.reduce_to_name(ion, halide, cation, unique=True, sort=True)
                simulation_names = {
                    "MD_perov_solv_L": "P%s_S%s_l" % (perovskite, solvent),
                    "MD_perov_solv_S": "P%s_S%s_N%d_s" % (perovskite, solvent, num_solvents),
                    "MD_solv_L": "S%s_l" % solvent,
                    "MD_solv_S": "S%s_N%d_s" % (solvent, num_solvents),
                    "DFT_perov": "P%s_R%d" % (perovskite, route_lvl),
                    "DFT_perov_solv": "P%s_S%s_N%d_R%d" % (perovskite, solvent, num_solvents, route_lvl),
                    "DFT_solv": "S%s_N%d_R%d" % (solvent, num_solvents, route_lvl)
                }
                # Convert names to aliases so we can avoid errors with long simulation names
                for k, v in simulation_names.items():
                    simulation_names[k] = parser.simname_to_alias(v)
                E_tot = orca.read(simulation_names["DFT_perov_solv"]).energy
                E_solv = orca.read(simulation_names["DFT_solv"]).energy
                E_perov = orca.read(simulation_names["DFT_perov"]).energy

                return units.convert_energy("Ha", unit, E_tot - E_solv - E_perov)

        # Attach the function
        running_job.result = types.MethodType(result, running_job)
        running_job.get_result = types.MethodType(result, running_job)

        return running_job
    else:
        return run_get_binding_energy(halide, cation, solvent, ion,
                                      route_lvl, unit, num_solvents,
                                      R_cutoff, rerun, run_sandboxed,
                                      queue, nprocs, xhost, priority=priority, verbose=verbose,
                                      walltime_orca=walltime_orca, override_seed=override_seed,
                                      lammps_queue=lammps_queue, lammps_nprocs=lammps_nprocs)

