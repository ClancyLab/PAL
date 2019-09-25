import os
from squid import jobs

njobs = range(2, 100)

##############################################################################

def run(sffx, nsample=1000):
    # Run misoKG Benchmarks
    script = '''
from run_misokg import run_misokg
run_misokg(
    $INDEX$,
    sffx="$SFFX$",
    SAMPLE_DOMAIN=$NSAMPLE$
)
    '''.strip()
    script = script.replace("$SFFX$", sffx)
    script = script.replace("$NSAMPLE$", str(nsample))

    for i in njobs:
        fptr = open("%d_%d_%s.py" % (i, nsample, sffx), 'w')
        fptr.write(script.replace("$INDEX$", str(i)))
        fptr.close()
    
        jobs.pysub(
            "%d_%d_%s" % (i, nsample, sffx),
            nprocs=2,
            queue="long",
            priority=10
        )

def run_ei(nsample=1000):
    script = '''
from run_ei import run_ei
run_ei(
    $INDEX$,
    SAMPLE_DOMAIN=$NSAMPLE$
)
'''.strip()
    script = script.replace("$NSAMPLE$", str(nsample))

    for i in njobs:
        fptr = open("%d_%d_ei.py" % (i, nsample), 'w')
        fptr.write(script.replace("$INDEX$", str(i)))
        fptr.close()
    
        jobs.pysub(
            "%d_%d_ei" % (i, nsample),
            nprocs=2,
            queue="long",
            priority=10
        )


#for sample in [100, 200, 500, 1000]:
for sample in [100, 200]:
    FOLDER = "RNS%d" % sample
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    run("misokg", nsample=sample)
    run("bdpc", nsample=sample)
    run("bidpc", nsample=sample)
    run("bvl", nsample=sample)
    run("bsvl", nsample=sample)
    run("bI", nsample=sample)
    run("bu", nsample=sample)
    run_ei(nsample=sample)

