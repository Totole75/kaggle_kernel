import os

import tools
from kernel import *
from algorithm import *
from tools import *

from sklearn import svm

folder_name = "data"

result_list = []
success_percentage = []

k_list = [6, 8, 2]
lmb_list = [0.45, 0.3, 0.17]
C_list = [1000, 100000, 10]
training_nb = 2000

for file_idx in range(3):
    train_file_name = "Xtr" + str(file_idx) + "_mat100.csv"
    train_str_file_name = "Xtr" + str(file_idx) + ".csv"
    result_file_name = "Ytr"+ str(file_idx) +".csv"
    test_file_name = "Xte" + str(file_idx) + "_mat100.csv"
    test_str_file_name = "Xte" + str(file_idx) + ".csv"

    train_array = tools.read_file(os.path.join(folder_name, train_file_name))
    str_train_array = tools.read_file(os.path.join(folder_name, train_str_file_name))
    label_array = tools.read_file(os.path.join(folder_name, result_file_name))
    test_array = tools.read_file(os.path.join(folder_name, test_file_name))
    str_test_array = tools.read_file(os.path.join(folder_name, test_str_file_name))
    str_array = np.concatenate((str_train_array, str_test_array))

    #######################################
    ### This is where you put your code ###
    #######################################

    subst_k, subst_lmb = k_list[file_idx], lmb_list[file_idx]

    import_ker = imported_kernel(str_array, training_nb, load_path="save_kernel\\substring_ker_file"+str(file_idx)+".npy")

    # sigma = 0.01
    # # lambda_reg = 1
    # lambda_svm = 0.1
    # gauss_ker = gaussian_kernel(train_array, sigma)
    # svm = SVM(label_array, lambda_svm, gauss_ker)
    # alpha_svm = svm.optimize_alpha()
    # predictions = gauss_ker.predict(test_array, alpha_svm)
    # test_classes = []
    # for i in range(len(test_array)):
    #     test_classes.append(0 if predictions[0, i] < 0 else 1)
    # test_classes = np.array(test_classes)
    # print("percentage of ones: ", np.mean(test_classes))

    print("svm")
    C_svm = C_list[file_idx]

    train_ker = import_ker.kernel_array[:training_nb, :][:, :training_nb]
    test_ker = import_ker.kernel_array[:training_nb, :][:, training_nb:]
    # Dividing by max to prevent numerical problems
    train_ker = train_ker / import_ker.kernel_array.max()
    test_ker = test_ker / import_ker.kernel_array.max()
    #print(train_ker, test_ker)

    clf = svm.SVC(kernel='precomputed', C=C_svm)
    clf.get_params()
    clf.fit(train_ker, label_array)
    
    predictions = clf.predict(test_ker.T)
    #print(predictions)
    test_classes = []
    for i in range(len(predictions)):
        test_classes.append(0 if predictions[i] < 0 else 1)
    test_classes = np.array(test_classes)
    print("percentage of ones: ", np.mean(test_classes))

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
