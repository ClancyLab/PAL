import os
import time
from squid.jobs import pysub


def fullreplace(book, word, newword):
    while word in book:
        book = book.replace(str(word), str(newword))
    return book


script = '''
from run_simple_$SIM import run_$SIM
run_$SIM($INDEX)
'''

if not os.path.exists("data_dumps"):
    os.mkdir("data_dumps")

UPPER = 10
RANGE = range(UPPER)

for sim in ["misokg"]:
#for sim in ["misokg", "ei", "kg"]:
    for i in RANGE:
        local = fullreplace(script, "$SIM", sim)
        local = fullreplace(local, "$INDEX", i)

        fname = "%d_%s.py" % (i, sim)

        fptr = open(fname, 'w')
        fptr.write(local)
        fptr.close()

        pysub(fname, queue="long", nprocs="4", use_mpi=False)
        time.sleep(0.1)
