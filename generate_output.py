import os

import tools
from kernel import *
from algorithm import *
from tools import *

folder_name = "data"

result_list = []

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

    lin_ker = linear_kernel(train_array)
    clusters = kernel_kmeans(lin_ker, 50, 200)
    test_classes = cluster_test(clusters, lin_ker,
                                label_array, test_array)

    #########################################
    ### Put your code before this comment ###
    #########################################    
    result_list.append(test_classes)

tools.write_output(result_list[0], 
                   result_list[1], 
                   result_list[2],
                   "result.csv")