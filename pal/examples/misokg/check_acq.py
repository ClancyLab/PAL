j = range(10)
i = None
folder = "data_dumps"
#folder = "released_rho_data_dumps"

reset = i is None
for RUN_INDEX in j:
    if reset:
        i = None
    
    is0 = open("%s/%d_acq_misokg.dat_IS0" % (folder, RUN_INDEX)).read().strip().split("\n")
    is1 = open("%s/%d_acq_misokg.dat_IS1" % (folder, RUN_INDEX)).read().strip().split("\n")

    sig0 = open("%s/%d_sig_misokg.dat_IS0" % (folder, RUN_INDEX)).read().strip().split("\n")
    sig1 = open("%s/%d_sig_misokg.dat_IS1" % (folder, RUN_INDEX)).read().strip().split("\n")
    
    if i is None:
        i = min([len(is0), len(is1), len(sig0), len(sig1)]) - 1
        if i == -1:
            i = 0
    
    if not isinstance(i, list):
        i = [i]
    
    for k in i:
        l_is0 = [float(v) for v in is0[k].strip().split()]
        l_is1 = [float(v) for v in is1[k].strip().split()]

        l_sig0 = [float(v) for v in sig0[k].strip().split()]
        l_sig1 = [float(v) for v in sig1[k].strip().split()]
        
        print "RUN_INDEX = %d, i = %d" % (RUN_INDEX, k)
        print "IS0: ", min(l_is0), max(l_is0), min(l_sig0), max(l_sig0)
        print "IS1: ", min(l_is1), max(l_is1), min(l_sig1), max(l_sig1)

#        if j == i[-1]:
#            print l_sig0
#            print l_sig1
