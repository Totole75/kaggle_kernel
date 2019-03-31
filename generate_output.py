import os

import tools
from kernel import *
from algorithm import *
from tools import *

from sklearn import svm

folder_name = "data"

result_list = []
success_percentage = []

lmb_list = [5e-6, 5e-9, 5e-4]
training_nb = 2000
multiplier = [1e16, 1e4, 1]

for file_idx in range(3):
    result_file_name = "Ytr"+ str(file_idx) +".csv"
    label_array = tools.read_file(os.path.join(folder_name, result_file_name))

    #######################################
    ### This is where you put your code ###
    #######################################

    name_file_training = 'substring_ker_file' + str(file_idx)
    data_array = multiplier[file_idx] * np.load(os.path.join(folder_name, name_file_training + '.npy'))
    kernel = imported_kernel(data_array, training_nb, center_kernel=True)

    print("svm")

    svm = SVM(label_array, lmb_list[file_idx], kernel)
    alpha = svm.optimize_alpha()
    predictions = kernel.predict(None, alpha)
    predictions.reshape((1, -1))
    test_classes = []
    for i in range(predictions.shape[1]):
        test_classes.append(0 if predictions[0, i] < 0 else 1)
    test_classes = np.array(test_classes)
    print("percentage of ones: ", np.mean(test_classes))

    #########################################
    ### Put your code before this comment ###
    #########################################    
    result_list.append(test_classes)

tools.write_output(result_list[0], 
                   result_list[1], 
                   result_list[2],
                   "result.csv")
