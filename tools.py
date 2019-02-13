import os
import numpy as np

def read_file(path_file):
    """read any file and return the right array"""
    with open(path_file, 'r') as f:
        L = f.readlines()
        if len(L[0]) == 9:
            #Y file
            matrix = np.zeros(len(L)-1)
            for index, l in enumerate(L):
                if index > 0:
                    matrix[index-1] = int(l.split(',')[1])
        elif len(L[0]) == 7:
            #X file
            matrix = []
            for index, l in enumerate(L):
                if index > 0:
                    matrix.append(l.split(',')[1])
        elif len(L[0]) > 100:
            #X_mat100 file
            matrix = np.zeros((len(L),100))
            for index, l in enumerate(L):
                matrix[index, :] = list(map(float, l.split(" ")))
        else:
            assert('ERROR')
    return(matrix)
    
#folder_name = 'data'
#file_list = os.listdir(folder_name)

#for file in file_list:
#    matrix = read_file(os.path.join(folder_name, file))

# file_feat = 'Xtr1_mat100.csv'
# feat = read_file(os.path.join(folder_name, file))
# file_label = 'Ytr1.csv'
# label = read_file(os.path.join(folder_name, file))
