import os
import time

import tools
from kernel import *
from algorithm import *

from sklearn.cluster import KMeans
from sklearn import svm

folder_name = "data"

result_list = []
success_percentage = []

k_list = [6, 8, 2]
lmb_list = [0.45, 0.3, 0.17]
C_list = [100, 100000, 10]

for file_idx in range(3):
    train_file_name = "Xtr" + str(file_idx) + "_mat100.csv"
    train_str_file_name = "Xtr" + str(file_idx) + ".csv"
    result_file_name = "Ytr"+ str(file_idx) +".csv"

    training_percentage = 0.8
    data_array = tools.read_file(os.path.join(folder_name, train_file_name))
    str_data_array = tools.read_file(os.path.join(folder_name, train_str_file_name))
    result_array = tools.read_file(os.path.join(folder_name, result_file_name))
    training_nb = round(training_percentage*data_array.shape[0])

    training_array = data_array[:training_nb, :]
    str_training_array = str_data_array[:training_nb, :]
    training_labels = result_array[:training_nb]

    test_array = data_array[training_nb:, :]
    str_test_array = str_data_array[training_nb:, :]
    test_labels = result_array[training_nb:]

    begin_time = time.time()
    #######################################
    ### This is where you put your code ###
    #######################################

    sigma = 0.01
    # lambda_reg = 1
    lambda_svm = 0.1
    lambda_log = 0.5
    gauss_ker = gaussian_kernel(data_array, training_nb, sigma)
    lin_ker = linear_kernel(data_array, training_nb)

    subst_k, subst_lmb = k_list[file_idx], lmb_list[file_idx]
    #subst_ker = substring_kernel(str_data_array, training_nb, subst_k, subst_lmb)
    # Enter the right path to load your array if you are using imported_kernel
    # Warning : don't change parameters if you use npy files and not regenerate them with same params too
    #import_ker = imported_kernel(str_data_array, training_nb, load_path="")
    
    # svm = SVM(training_labels, lambda_svm, gauss_ker)
    # alpha_svm = svm.optimize_alpha()
    # predictions = gauss_ker.predict(test_array, alpha_svm)
    # test_classes = []
    # for i in range(len(test_labels)):
    #     test_classes.append(0 if predictions[0, i] < 0 else 1)
    # test_classes = np.array(test_classes)

    # alpha_ridge = ridge_regression(gaus_ker.kernel_array, training_labels, lambda_reg)
    # alpha_lin = ridge_regression(lin_ker.kernel_array, training_labels, lambda_reg)

    # log_reg = logistic_regression(training_labels, lambda_log, subst_ker)
    # logistic_alpha = log_reg.optimize_alpha()
    # predictions = subst_ker.predict(str_test_array, logistic_alpha)
    # test_classes = []
    # for i in range(len(test_labels)):
    #     test_classes.append(0 if predictions[0, i] < 0 else 1)
    # test_classes = np.array(test_classes)
    # print("percentage of ones: ", np.mean(test_classes))

    print("svm")
    C_svm = C_list[file_idx]

    train_ker = lin_ker.kernel_array[:training_nb, :][:, :training_nb]
    test_ker = lin_ker.kernel_array[:training_nb, :][:, training_nb:]
    # Dividing by max to prevent numerical problems
    train_ker = train_ker / lin_ker.kernel_array.max()
    test_ker = test_ker / lin_ker.kernel_array.max()
    print(train_ker, test_ker)

    clf = svm.SVC(kernel='precomputed', C=C_svm)
    print("fit")
    clf.get_params()
    clf.fit(train_ker, training_labels)
    
    predictions = clf.predict(test_ker.T)
    print(predictions)
    test_classes = []
    for i in range(len(test_labels)):
        test_classes.append(0 if predictions[i] < 0 else 1)
    test_classes = np.array(test_classes)
    print("percentage of ones: ", np.mean(test_classes))

    # prediction_ridge = gaus_ker.predict(test_array, alpha_ridge)

    # kmeans_clustering = Kmeans(lin_ker)
    # kmeans_clustering.create_clusters(cluster_nb=100, restart_nb=20)
    # test_classes = kmeans_clustering.predict(training_labels)

    #########################################
    ### Put your code before this comment ###
    #########################################
    end_time = time.time()
    print("File " + str(file_idx) + " took : " + str(end_time-begin_time) + " seconds")
    
    result_list.append(test_classes)
    success_percentage.append(np.mean(np.equal(test_classes, ((test_labels+1)/2).astype(int))))
    print("Result for file " + str(file_idx) + " : " + str(success_percentage[file_idx]))