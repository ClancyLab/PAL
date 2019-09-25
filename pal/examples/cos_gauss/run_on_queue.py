import os
from squid import jobs

njobs = range(100)

##############################################################################

if not os.path.exists("data_dumps"):
    os.mkdir("data_dumps")

def run(name, sffx, scaled=False, loose=False, very_loose=False, use_MAP=False, upper=1.0, run_standard_misokg=False, unitary=False, use_miso=False):
    # Run misoKG Benchmarks
    script = '''
from run_misokg import run_misokg
run_misokg(
    $INDEX$,
    sffx="$SFFX$",
    scaled=$SCALED$,
    loose=$LOOSE$,
    very_loose=$VERY_LOOSE$,
    use_MAP=$USE_MAP$,
    upper=$UPPER$,
    unitary=$RUN_UNITARY$,
    use_miso=$RSM$
)
    '''.strip()
    script = script.replace("$SFFX$", sffx)
    script = script.replace("$SCALED$", str(scaled))
    script = script.replace("$LOOSE$", str(loose))
    script = script.replace("$VERY_LOOSE$", str(very_loose))
    script = script.replace("$USE_MAP$", str(use_MAP))
    script = script.replace("$UPPER$", str(upper))
    script = script.replace("$RSM$", str(use_miso))
    script = script.replace("$RUN_UNITARY$", str(unitary))
    
    for i in njobs:
        fptr = open("%d_%s.py" % (i, name), 'w')
        fptr.write(script.replace("$INDEX$", str(i)))
        fptr.close()
    
        jobs.pysub(
            "%d_%s" % (i, name),
            nprocs=2,
            queue="long",
            priority=10
        )

run("rosen", "misokg", use_miso=True)
run("rosen_bu", "bunit", unitary=1.0 - 1E-6)
run("rosen_pc", "pc", unitary=0.79)
run("rosen_vl", "vl", scaled=False, very_loose=True)
run("rosen_svl", "svl", scaled=True, very_loose=True)

