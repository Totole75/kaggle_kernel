import os

import tools
from kernel import *
from algorithm import *

folder_name = "data"
file_name = "Xtr0_mat100.csv"
result_file_name = "Ytr0.csv"

data_array = tools.read_file(os.path.join(folder_name, file_name))
result_array = tools.read_file(os.path.join(folder_name, result_file_name))

print(result_array)

lin_ker = linear_kernel(data_array)
print(lin_ker.kernel_array.shape)