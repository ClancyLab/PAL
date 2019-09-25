import os
import copy
import time
import numpy as np

# From Clancelot
from squid import orca
from squid import units
from squid import files
from squid import geometry
from squid import sysconst
from squid import constants
from squid import structures

# From PAL
import pal.log.database as logDB
import pal.constants.world as WORLD_CONSTS

read_log, dump_log = logDB.read_pickle, logDB.dump_pickle

SLEEPER = 60

def _generate_ion_halide(halide, ion="Pb"):
    """
    A function to generate atomic coordinates of an ion and three halides for
    a perovskite monomer.

    **Parameters**

        halide: *list, str or str*
            The halides to use.
        ion: *str, optional*
            What ion to use.

    **Returns**

        ionX3: *Molecule*
            A molecule object holding the perovskite ion + halide atoms.

    """
    ionX3 = structures.Molecule([structures.Atom(ion, 0, 0, 0)])
    if type(halide) is str:
        halide = [halide, halide, halide]

    def vdw(y):
        return constants.PERIODIC_TABLE[units.elem_s2i(y)]['vdw_r']

    for x in halide:
        v = vdw(x)
        ionX3.atoms.append(structures.Atom(x, v, 0, 0.5 * v))
        R = geometry.rotation_matrix([0, 0, 1], 120, units="deg")
        ionX3.rotate(R)
    return ionX3


def generate_ion_halide_cation(halide, cation, fname,
                               route, extra_section,
                               ion="Pb", run_opt=True, priority=None,
                               walltime="3-0:0:0", procs=2):
    """
    Check if a perovskite monomer system already exists. If so, then read in
    the CML and return it. If not, then generate the system, save it, and
    return the generated system.

    **Parameters**

        halide: *list, str or str*
            The halide combination in the perovskite. If a string is passed,
            all three are assumed to be the same
        cation: *str*
            The cation in the perovskite.
        ion: *str, optional*
            The ion to use within this perovskite.
        run_opt: *bool, optional*
            Whether to geometry optimize the system if it did not exist.
        priority: *int, optional*
            A queue priority to ensure the subjobs run.  This may be necessary if
            many simulations are run in parallel and you are worried about locking
            the queue with too many parent jobs.

    **Returns**

        system: *Molecule*
            A molecule object holding the perovskite monomer system.
    """

    if os.path.exists(WORLD_CONSTS.STRUCTURES_PATH + fname + ".cml"):
        system = structures.Molecule(files.read_cml(WORLD_CONSTS.STRUCTURES_PATH + fname + ".cml",
                                                    test_charges=False, default_angles=WORLD_CONSTS.default_angles,
                                                    allow_errors=True)[0])
        return system

    def vdw(y):
        return constants.PERIODIC_TABLE[units.elem_s2i(y)]['vdw_r']

    # Get the ionX3 system
    ionX3 = _generate_ion_halide(halide, ion=ion)

    # Get the cation from the cml file
    atoms, bonds, _, _ = files.read_cml(WORLD_CONSTS.STRUCTURES_PATH + cation + ".cml",
                                        test_charges=False, default_angles=WORLD_CONSTS.default_angles,
                                        allow_errors=True)

    system = structures.Molecule(atoms)

    # Align along X axis
    system.atoms = geometry.align_centroid(system.atoms)[0]
    # Rotate to Z axis
    # NOTE! In case of FA, we want flat so only translate to origin instead
    elems = [a.element for a in system.atoms]
    if cation == "FA":
        system.translate(system.get_center_of_mass())
    else:
        R = geometry.rotation_matrix([0, 1, 0], 90, units="deg")
        system.rotate(R)
    # If N and C in system, ensure N is below C (closer to Pb)
    if "N" in elems and "C" in elems:
        N_index = [i for i, a in enumerate(system.atoms)
                   if a.element == "N"][0]
        C_index = [i for i, a in enumerate(system.atoms)
                   if a.element == "C"][0]
        if system.atoms[N_index].z > system.atoms[C_index].z:
            # Flip if needed
            R = geometry.rotation_matrix([0, 1, 0], 180, units="deg")
            system.rotate(R)

    # Offset system so lowest point is at 0 in the z dir
    z_offset = min([a.z for a in system.atoms]) * -1
    system.translate([0, 0, z_offset])

    # Add to the ionX3 system with an offset of vdw(Pb)
    system.translate([0, 0, vdw(ion)])
    system.atoms += ionX3.atoms

    # Run a geometry optimization of this system
    if run_opt:
        ionXY = orca.job(fname, route,
                         atoms=system.atoms,
                         extra_section=extra_section,
                         redundancy=True,
                         queue=sysconst.default_queue,
                         priority=priority,
                         walltime=walltime,
                         procs=procs)
        time.sleep(SLEEPER)
        ionXY.wait()
        time.sleep(SLEEPER)
        results = orca.read(fname)
        new_pos = results.atoms
        for a, b in zip(system.atoms, new_pos):
            a.x, a.y, a.z = [b.x, b.y, b.z]

        if not hasattr(ionXY, "redundancy"):
            dump_log(results, fname)

    # Set OPLS types
    for i, a in enumerate(system.atoms):
        a.index = i + 1
        if a.element in WORLD_CONSTS.atom_types and a.type is None:
            a.type = structures.Struct()
            a.type.__dict__.update(WORLD_CONSTS.atom_types[a.element])
            a.type_index = a.type.index

    # Write cml file so we don't re-generate, and return system
    files.write_cml(system, bonds=bonds, name=WORLD_CONSTS.STRUCTURES_PATH + fname + ".cml")
    return system


def strip_solvents(system, R_cutoff, num_solvents, ion="Pb", polar_atom="O", solute=None, no_solute=False):
    '''
    **Parameters**

        solute: *structures.Molecule, optional*
            This is used if we want to identify the solute by a molecule object instead of an ion.

        no_solute: *bool, optional*
            This is used if, when stripping molecules, we want to only grab solvents.
    '''

    if solute is not None:
        raise Exception("Ideally we would also allow solute isolation from procrustes comparison.  But not implemented yet.")

    # Step 1 - Isolate the solute and solvents
    if no_solute:
        # Note, we grab a solvent near the origin so it's less likely to cross a periodic
        # boundary.  We do this instead of manually handling perodic boundaries by offsetting
        # atoms by the box size because if we use NPT, the box size itself changes!  The
        # alternative is to unwrap atomic coordinates in lammps itself before reading it in.
        index = [
            (np.linalg.norm(np.array(a.get_center_of_geometry()) - np.array([0, 0, 0])), i)
            for i, a in enumerate(system.molecules)
        ]
        index = sorted(index, key=lambda x: x[0])[0][1]
        solute = system.molecules[index]
        solute_ion = solute.atoms[0]
    else:
        solute = [m for m in system.molecules if ion in [a.element for a in m.atoms]][0]
        solute_ion = [a for a in solute.atoms if a.element == ion][0]
    solvents = [m for m in system.molecules if m is not solute]

    # Step 2 - Get a list of all polar atoms in the molecules
    if polar_atom is None:
        polar_atom_list = [[a for a in m.atoms] for m in solvents]
    else:
        if not isinstance(polar_atom, list):
            polar_atom = [polar_atom]
        polar_atom_list = [[a for a in m.atoms if a.element in polar_atom] for m in solvents]

    # Step 3 - Get closest relative distances to solute ion
    proximity = [
        (min([geometry.dist(solute_ion, a) for a in polar_atom_list[i]]), m) for i, m in enumerate(solvents)
    ]
    proximity = sorted(proximity, key=lambda x: x[0])

    # Step 4 - Take into account R_cutoff
    try:
        index = [int(d > R_cutoff) for d, _ in proximity].index(1)
        proximity = proximity[:index]
    except ValueError:
        # In this case, we are collecting everything
        pass

    # Step 5 - Take into account num_solvents
    if len(proximity) > num_solvents:
        proximity = proximity[:num_solvents - int(no_solute)]

    # Step 6 - Generate new system and return it
    # NOTE! ORDER MATTERS! [(None, solute)] + proximity is this way because
    # we assume that the first molecule is the solute that needs to be made
    # imobile.
    new_system = structures.System(box_size=(25, 25, 25))
    for _, m in [(None, solute)] + proximity:
        for i, a in enumerate(m.atoms):
            a.index = i + 1
        new_system.add(copy.deepcopy(m))

    # Assign types that otherwise don't exist
    for a in new_system.atoms:
        if a.element in WORLD_CONSTS.atom_types and a.type is None:
            a.type = structures.Struct()
            a.type.__dict__.update(WORLD_CONSTS.atom_types[a.element])
            a.type_index = a.type.index

    new_system.set_types()

    return new_system


def _test_generate_ion_halide_cation():
    obj = generate_ion_halide_cation(
        ["Cl", "Br", "Br"],
        "Cs",
        "CsPbBrBrCl_0",
        WORLD_CONSTS.default_routes[0],
        WORLD_CONSTS.extra_section,
        ion="Pb",
        run_opt=False).atoms
    saved = [
        structures.Atom("Cs", 0.0, 0.0, 2.5299999999999998),
        structures.Atom("Pb", 0.0, 0.0, 0.0),
        structures.Atom("Cl", 2.0499999999999998, -3.3306690738754696e-15, 1.0249999999999999),
        structures.Atom("Br", -1.050000000000002, -1.8186533479473201, 1.05),
        structures.Atom("Br", -1.0499999999999989, 1.8186533479473219, 1.05),
    ]
    TOLERANCE = 1E-6
    return geometry.motion_per_frame([obj, saved])[1] < TOLERANCE


def run_unit_tests():
    assert _test_generate_ion_halide_cation(), "pal.utils.generate_systems.generate_ion_halide_cation() failed."


if __name__ == "__main__":
    run_unit_tests()

