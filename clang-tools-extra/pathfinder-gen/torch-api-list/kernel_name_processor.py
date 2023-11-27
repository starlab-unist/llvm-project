block_list = [ # cuda kernels
  "_fft_c2c_cufft",
  "_fft_r2c_cufft",
  "_fft_c2r_cufft",
]

f = open("_ops_to_kernels.txt", "r") #https://gitlab.com/brown-ssl/ivysyn/-/blob/main/src/ivysyn/pytorch/ops_to_kernels.txt
kernels = []
for line in f.readlines():
  kernels += line.split(" ")[1].split(",")
f2 = open("kernel", "w")
for kernel in kernels:
  kernel = kernel.strip()
  if kernel not in block_list:
    f2.write("at::native::" + kernel.strip()+"\n")
