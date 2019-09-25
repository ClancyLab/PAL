import os
import cPickle as pickle
from pal.constants.world import DB_FPTR, BINDING_ENERGY_DATABASE


def dump_pickle(obj, name, warn=False):
    '''
    A function to dump objects individually to a pickle file in some given
    database folder.

    **Parameters**

        obj: *object*
            The object to be saved in a pickle file.

        name: *str*
            The name of the file.

    **Returns**

        None
    '''
    if warn and os.path.exists(BINDING_ENERGY_DATABASE + name):
        print("WARNING! WHEN SAVING TO DATABASE, OVERWRITTING!")
    pickle.dump(obj, open(BINDING_ENERGY_DATABASE + name, 'w'))


def read_pickle(name):
    '''
    A function to read objects from a pickle file in some givin databse
    folder.

    **Parameters**

        name: *str*
            The name of the file.

    **Returns**

        obj: *object*
            The stored object.
    '''
    if not os.path.isfile(BINDING_ENERGY_DATABASE + name):
        return None
    return pickle.load(open(BINDING_ENERGY_DATABASE + name, 'r'))


def dump_to_db(obj, name, warn=False):
    '''
    A function to dump objects to some database, which is a dictionary saved
    as a pickle file.

    **Parameters**

        obj: *object*
            The object to be saved.

        name: *str*
            The key associated with the object.

    **Returns**

        None
    '''
    # If the database doesn't exist, generate a new one!
    if not os.path.exists(DB_FPTR):
        db = {}
        print("Warning - No database file exists at %s. Generating new." % DB_FPTR)
    else:
        db = pickle.load(open(DB_FPTR, 'r'))
    if warn and name in db:
        print("WARNING! WHEN SAVING TO DATABASE, OVERWRITTING!")
    db[name] = obj

    pickle.dump(db, open(DB_FPTR, 'w'))


def read_from_db(name):
    try:
        db = pickle.load(open(DB_FPTR, 'r'))
    except IOError:
        # If no db file exists yet, return None
        return None
    if name not in db:
        return None
    return db[name]
