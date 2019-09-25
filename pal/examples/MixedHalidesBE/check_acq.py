i = None
RUN_INDEX = 0

i = range(50)

is0 = open("data_dumps/%d_acq_misokg.dat_IS0" % RUN_INDEX).read().strip().split("\n")
is1 = open("data_dumps/%d_acq_misokg.dat_IS1" % RUN_INDEX).read().strip().split("\n")

if i is None:
    i = min(len(is0), len(is1)) - 1

if not isinstance(i, list):
    i = [i]

for j in i:
    l_is0 = [float(v) for v in is0[j].strip().split()]
    l_is1 = [float(v) for v in is1[j].strip().split()]
    
    print "RUN_INDEX = %d, i = %d" % (RUN_INDEX, j)
    print "IS0: ", min(l_is0), max(l_is0)
    print "IS1: ", min(l_is1), max(l_is1)
