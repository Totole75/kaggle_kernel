import os

import tools
from kernel import *
from algorithm import *

folder_name = "data"
file_name = "Xtr0_mat100.csv"
result_file_name = "Ytr0.csv"
test_file_name = "Xte0_mat100.csv"

training_percentage = 0.9
data_array = tools.read_file(os.path.join(folder_name, file_name))
result_array = tools.read_file(os.path.join(folder_name, result_file_name))
training_nb = round(training_percentage*data_array.shape[0])
training_array = data_array[:training_nb, :]
training_results = result_array[:training_nb]
test_array = data_array[training_nb:, :]
test_results = result_array[training_nb:]

sigma = 1
# lambda_reg = 1
gaus_ker = gaussian_kernel(training_array, sigma)
lin_ker = linear_kernel(training_array)

# alpha_ridge = ridge_regression(gaus_ker.kernel_array, training_results, lambda_reg)
# alpha_lin = ridge_regression(lin_ker.kernel_array, training_results, lambda_reg)

# prediction_ridge = gaus_ker.predict(test_array, alpha_ridge)


clusters = kernel_kmeans(lin_ker, 50, 200)
test_classes = cluster_test(clusters, lin_ker,
                            training_results,
                            test_array)

print(np.equal(test_classes, test_results).mean())


# data_array = np.array([10,12,-5,8,-6,-4]).reshape((6, 1))
# k = linear_kernel(data_array)
# S = SVM([0,0,1,0,1,1], 0.0001, k)
# alpha = S.optimize_alpha()
# print("alpha", alpha)
# print(S.test(alpha, data_array, [15,-1], np.array([0,1])))
