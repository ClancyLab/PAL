Physical Analytics pipeLine (PAL)
=====================================
An automated interface to DFT and MD codes for specific properties.

If you wish to calculate the MBO of a given solvent to perovskite composition,
for example the average O-H MBO in water to PbI3MA, you can do so with the
following command:

.. code-block:: python

    MBO = pal.get_MBO("Cl", "Cs", "WATER",
                      ion="Pb",
                      route_lvl=0,
                      on_queue=False,
                      criteria=[["O", "H"]],
                      queue="batch", nprocs=2, xhost=None)

Similarly, for the UMBO you can simply do:

.. code-block:: python

    UMBO = pal.get_UMBO("Cl", "Cs", "WATER",
                        ion="Pb",
                        route_lvl=0,
                        on_queue=False,
                        criteria=[["O", "H"]],
                        offset=1.0,
                        queue="batch", nprocs=2, xhost=None)

Note the offset value here.  This simply gives the Formal Bond Order (FBO).

Now, for binding energy calculations between a solvent and perovskite, the
following works:

.. code-block:: python

    E = pal.get_enthalpy_solvation("Cl", "Cs", "WATER",
                                   ion="Pb",
                                   route_lvl=0,
                                   unit="kT_300",
                                   num_solvents=4,
                                   R_cutoff=2000.0,
                                   on_queue=False,
                                   queue="batch", nprocs=2, xhost=None)

Note, this will calculate the binding energy of four water molecules!

If you wish to use a mixed solvent, the following can be done:

.. code-block:: python

    E = pal.get_enthalpy_solvation("Cl", "Cs", {"WATER": 0.75, "GBL": 0.25},
                                   ion="Pb",
                                   route_lvl=0,
                                   unit="kT_300",
                                   num_solvents=4,
                                   R_cutoff=2000.0,
                                   on_queue=False,
                                   queue="batch", nprocs=2, xhost=None)

This specifies a 3:1 ratio of water to GBL.  By setting the number of solvents
to 4, we get exactly 3 water molecules and one GBL molecule.

Finally, if one wishes to calculate any of the above properties for a mixed
halide system, they need only pass a list of halides instead.  For instance:

.. code-block:: python

    E = pal.get_enthalpy_solvation(["Cl", "Cl", "Br"], "MA", "DMSO",
                                   ion="Pb",
                                   route_lvl=0,
                                   unit="kT_300",
                                   num_solvents=1,
                                   R_cutoff=2000.0,
                                   on_queue=False,
                                   queue="batch", nprocs=2, xhost=None)

H_solv
--------------------

.. automodule:: H_solv
    :members:

MBO
--------------------

.. automodule:: MBO
    :members:

UMBO
--------------------

.. automodule:: UMBO
    :members:

