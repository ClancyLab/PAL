import os

#raise Exception("STOP!")

for f in os.listdir("."):
    if "_" not in f:
        continue
    chk_i = f.split("_")[0]
    if chk_i.isdigit():
        os.system("rm %s" % f)

