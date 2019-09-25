for i in range(100):
    data = open("%d_misokg.log" % i, 'r').read().strip().split("\r")
    data = [d.strip() for d in data][1:-1]
    fptr = open("data_dumps/%d_misokg.log" % i, 'w')
    fptr.write("\n".join(data))
    fptr.close()


