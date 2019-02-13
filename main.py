import os


import tools
from kernel import *
from algorithm import *

folder_name = "data"
file_name = "Xtr0_mat100.csv"
result_file_name = "Ytr0.csv"
test_file_name = "Xte0_mat100.csv"

data_array = tools.read_file(os.path.join(folder_name, file_name))
test_array = tools.read_file(os.path.join(folder_name, test_file_name))
label_array = tools.read_file(os.path.join(folder_name, result_file_name))

print(label_array)


print(lin_ker.kernel_array.shape)

sigma = 1
lambda_reg = 1
gaus_ker = gaussian_kernel(data_array, sigma)
lin_ker = linear_kernel(data_array)

alpha_ridge = ridge_regression(gaus_ker.kernel_array, label_array, lambda_reg)
alpha_lin = ridge_regression(lin_ker.kernel_array, label_array, lambda_reg)

prediction_ridge = gaus_ker.predict(test_array, alpha_ridge)