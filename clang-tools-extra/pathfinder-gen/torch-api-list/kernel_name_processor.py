f = open("_ops_to_kernels.txt", "r") #https://gitlab.com/brown-ssl/ivysyn/-/blob/main/src/ivysyn/pytorch/ops_to_kernels.txt
kernels = []
for line in f.readlines():
  kernels += line.split(" ")[1].split(",")
f2 = open("kernel", "w")
for kernel in kernels:
  f2.write("at::native::" + kernel.strip()+"\n")
