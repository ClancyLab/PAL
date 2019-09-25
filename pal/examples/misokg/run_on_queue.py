import os
import time
from squid.jobs import pysub


def fullreplace(book, word, newword):
    while word in book:
        book = book.replace(str(word), str(newword))
    return book


script = '''
from misokg_$SIM import run_misokg

run_misokg($INDEX)
'''

if not os.path.exists("data_dumps"):
    os.mkdir("data_dumps")

UPPER = 10
RANGE = range(UPPER)

for j, sim in enumerate(["diag_B_as_Ks", "diag_B_as_L", "scaled_Ks"]):
    for i in RANGE:
        print("Submitted %d_%s" % (i, sim))
        local = fullreplace(script, "$SIM", sim)
        local = fullreplace(local, "$INDEX", i + UPPER * j)

        fname = "%d_%s.py" % (i, sim)

        fptr = open(fname, 'w')
        fptr.write(local)
        fptr.close()

        if sim != "ei":
            pysub(fname, queue="shared", ntasks=4, walltime="20:00:00", use_mpi=False)
        else:
            pysub(fname, queue="shared", ntasks=2, walltime="6:00:00", use_mpi=False)
    time.sleep(2.0)
