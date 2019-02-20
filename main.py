import os

import tools
from kernel import *
from algorithm import *

from sklearn.cluster import KMeans


folder_name = "data"

result_list = []
success_percentage = []

for file_idx in range(3):
    train_file_name = "Xtr" + str(file_idx) + "_mat100.csv"
    result_file_name = "Ytr"+ str(file_idx) +".csv"

    training_percentage = 0.9
    data_array = tools.read_file(os.path.join(folder_name, train_file_name))
    result_array = tools.read_file(os.path.join(folder_name, result_file_name))
    training_nb = round(training_percentage*data_array.shape[0])
    training_array = data_array[:training_nb, :]
    training_labels = result_array[:training_nb]
    test_array = data_array[training_nb:, :]
    test_labels = result_array[training_nb:]

    #######################################
    ### This is where you put your code ###
    #######################################

    sigma = 0.01
    # lambda_reg = 1
    lambda_svm = 0.1
    gauss_ker = gaussian_kernel(training_array, sigma)
    lin_ker = linear_kernel(training_array)
    svm = SVM(training_labels, lambda_svm, gauss_ker)
    alpha_svm = svm.optimize_alpha()
    predictions = gauss_ker.predict(test_array, alpha_svm)
    test_classes = []
    for i in range(len(test_labels)):
        test_classes.append(0 if predictions[0, i] < 0 else 1)
    test_classes = np.array(test_classes)
    print("percentage of ones: ", np.mean(test_classes))

    # alpha_ridge = ridge_regression(gaus_ker.kernel_array, training_labels, lambda_reg)
    # alpha_lin = ridge_regression(lin_ker.kernel_array, training_labels, lambda_reg)

    # prediction_ridge = gaus_ker.predict(test_array, alpha_ridge)

    #kmeans_clustering = Kmeans(gauss_ker)
    #kmeans_clustering.create_clusters(cluster_nb=100, restart_nb=20)
    #test_classes = kmeans_clustering.predict(training_array, training_labels, test_array)

    #########################################
    ### Put your code before this comment ###
    #########################################
    
    result_list.append(test_classes)
    success_percentage.append(np.mean(np.equal(test_classes, ((test_labels+1)/2).astype(int))))
    print("Result for file " + str(file_idx) + " : " + str(success_percentage[file_idx]))

# data_array = np.array([10,12,-5,8,-6,-4]).reshape((6, 1))
# k = linear_kernel(data_array)
# S = SVM([0,0,1,0,1,1], 0.0001, k)
# alpha = S.optimize_alpha()
# print("alpha", alpha)
# print(S.test(alpha, data_array, [15,-1], np.array([0,1])))
