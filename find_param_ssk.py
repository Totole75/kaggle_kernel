"""
Code for testing different values of parameters for the substring kernel
"""

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

file_idx = 2

train_file_name = "Xtr" + str(file_idx) + "_mat100.csv"
train_str_file_name = "Xtr" + str(file_idx) + ".csv"
result_file_name = "Ytr"+ str(file_idx) +".csv"

data_array = tools.read_file(os.path.join(folder_name, train_file_name))
str_data_array = tools.read_file(os.path.join(folder_name, train_str_file_name))
result_array = tools.read_file(os.path.join(folder_name, result_file_name))

training_percentage = 0.2
test_percentage = 0.1
training_nb = round(training_percentage*data_array.shape[0])
test_nb = round(data_array.shape[0] - (test_percentage*data_array.shape[0]))
indices_to_keep = list(range(training_nb))+list(range(test_nb, data_array.shape[0]))
used_data_array = str_data_array[indices_to_keep, :]

print(training_nb, test_nb)

training_array = data_array[:training_nb, :]
str_training_array = str_data_array[:training_nb, :]
training_labels = result_array[:training_nb]

test_array = data_array[test_nb:, :]
str_test_array = str_data_array[test_nb:, :]
test_labels = result_array[test_nb:]
print(str_training_array.shape, str_test_array.shape)



begin_time = time.time()
#######################################
### This is where you put your code ###
#######################################

# k_values = range(2,9)
# lmb_values = np.arange(0.05, 0.4, 0.05)
# C_values = [10**i for i in range(1, 8)]
k_values = range(2,3)
lmb_values = np.arange(0.2, 0.3, 0.2)
C_values = [10**i for i in range(1, 10)]
# k_values = range(2,8)
# lmb_values = np.arange(0.05, 0.4, 0.05)
# C_values = [10**i for i in range(3, 5)]
# k_values = range(2,8)
# lmb_values = np.arange(0.02, 0.1, 0.01)
# C_values = [10**i for i in range(3, 6)]
# k_values = [3,6,7]
# lmb_values = np.arange(0.02, 0.1, 0.01)
# C_values = [10**i for i in range(3, 6)]
# k_values = range(2,13)
# lmb_values = [0.03]
# C_values = [1000]
# k_values = range(2,9)
# lmb_values = np.arange(0.1, 1, 0.1)
# C_values = [10**i for i in range(5, 8)]
# k_values = range(2,12)
# lmb_values = np.arange(0.45, 0.5, 1)
# C_values = [10**i for i in range(3,4)]

results = []
param = []

for subst_lmb in lmb_values:
    for C_svm in C_values:
        for subst_k in k_values:
            subst_ker = substring_kernel(used_data_array, training_nb, subst_k, subst_lmb)

            test_ker = subst_ker.kernel_array[:training_nb, :][:, training_nb:]

            clf = svm.SVC(kernel='precomputed', C=C_svm)
            clf.get_params()
            clf.fit(subst_ker.kernel_array[:training_nb, :][:, :training_nb]/subst_ker.kernel_array.max(), training_labels)

            predictions = clf.predict(test_ker.T/subst_ker.kernel_array.max())
            # print(predictions)
            test_classes = []
            for i in range(len(test_labels)):
                test_classes.append(0 if predictions[i] < 0 else 1)
            test_classes = np.array(test_classes)

            #########################################
            ### Put your code before this comment ###
            #########################################
            end_time = time.time()
            # print("File " + str(file_idx) + " took : " + str(end_time-begin_time) + " seconds")

            result_list.append(test_classes)
            success_percentage = np.mean(np.equal(test_classes, ((test_labels+1)/2).astype(int)))
            print("k " + str(subst_k) + " lmb " + str(subst_lmb.round(decimals=2)) + " C " + str(C_svm) + " File " + str(file_idx) + " mean_pred " + str(test_classes.mean()) + "    SCORE \t" + str(success_percentage))
            results.append(success_percentage)
            param.append("k " + str(subst_k) + " lmb " + str(subst_lmb.round(decimals=2)) + " C " + str(C_svm) + " File " + str(file_idx) + " mean_pred " + str(test_classes.mean()) + "    SCORE \t" + str(success_percentage))

            # Commands used to break the loop if the value of C is too low and causes all labels to be predicted at 1 or 0
            # if ((test_classes.mean() == 0) or (test_classes.mean() == 1)):
            #     break

print("fin")
results = np.array(results)
print(np.argmax(results), np.max(results))
print(param[np.argmax(results)])