import os

for fptr in os.listdir("./"):
    if fptr.split("_")[0].isdigit():
        os.system("rm %s" % fptr)

os.system("rm data_dumps/*")
os.system("rm imgs/*")
for f in ["sampled.dat", "tmp.log", "output_0.gif", "output_1.gif"]:
    if os.path.exists(f):
        os.system("rm %s" % f)
