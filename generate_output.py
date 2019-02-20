import os

import tools
from kernel import *
from algorithm import *
from tools import *

folder_name = "data"

result_list = []
success_percentage = []

for file_idx in range(3):
    train_file_name = "Xtr" + str(file_idx) + "_mat100.csv"
    result_file_name = "Ytr"+ str(file_idx) +".csv"
    test_file_name = "Xte" + str(file_idx) + "_mat100.csv"

    train_array = tools.read_file(os.path.join(folder_name, train_file_name))
    label_array = tools.read_file(os.path.join(folder_name, result_file_name))
    test_array = tools.read_file(os.path.join(folder_name, test_file_name))

    #######################################
    ### This is where you put your code ###
    #######################################
    sigma = 0.01
    # lambda_reg = 1
    lambda_svm = 0.1
    gauss_ker = gaussian_kernel(train_array, sigma)
    svm = SVM(label_array, lambda_svm, gauss_ker)
    alpha_svm = svm.optimize_alpha()
    predictions = gauss_ker.predict(test_array, alpha_svm)
    test_classes = []
    for i in range(len(test_array)):
        test_classes.append(0 if predictions[0, i] < 0 else 1)
    test_classes = np.array(test_classes)
    print("percentage of ones: ", np.mean(test_classes))

    #lin_ker = gaussian_kernel(train_array, 0.1)
    #lin_ker = linear_kernel(train_array)

    #########################################
    ### Put your code before this comment ###
    #########################################    
    result_list.append(test_classes)
    #success_percentage.append(np.mean(np.equal(test_classes, label_array)))

#success_percentage = np.mean(success_percentage)
#print("Pourcentage de reussite ", success_percentage)
tools.write_output(result_list[0], 
                   result_list[1], 
                   result_list[2],
                   "result.csv")
