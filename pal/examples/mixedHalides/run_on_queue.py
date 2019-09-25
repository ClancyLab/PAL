import os
import time
from squid.jobs import pysub


UPPER = 100
RANGE = range(UPPER)
QUEUE = "shared"
WALLTIME = "1-00:00:00"

def fullreplace(book, word, newword):
    while word in book:
        book = book.replace(str(word), str(newword))
    return book

script = '''
from run_simple_$SIM import run_$SIM
run_$SIM($INDEX, "$SFFX$", unitary=$UNITARY$)
'''
sim = "bonilla"
for index, unitary in enumerate([None, 1.0 - 1E-6, 0.83]):
    for i in RANGE:
        sffx = ["bl", "bu", "bpc"][index]

        local = fullreplace(script, "$SIM", sim)
        local = fullreplace(local, "$INDEX", i)
        local = fullreplace(local, "$UNITARY$", unitary)
        local = fullreplace(local, "$SFFX$", sffx)
    
        fname = "%d_%s_%s.py" % (i, sim, sffx)
    
        fptr = open(fname, 'w')
        fptr.write(local)
        fptr.close()
    
        pysub(fname, queue=QUEUE, nprocs="4", use_mpi=False, walltime=WALLTIME)
        time.sleep(0.1)

script = '''
from run_simple_$SIM import run_$SIM
run_$SIM($INDEX)
'''

if not os.path.exists("data_dumps"):
    os.mkdir("data_dumps")

for sim in ["ei", "kg"]:
    for i in RANGE:
        local = fullreplace(script, "$SIM", sim)
        local = fullreplace(local, "$INDEX", i)

        fname = "%d_%s.py" % (i, sim)

        fptr = open(fname, 'w')
        fptr.write(local)
        fptr.close()

        pysub(fname, queue=QUEUE, nprocs="4", use_mpi=False, walltime=WALLTIME)
        time.sleep(0.1)

