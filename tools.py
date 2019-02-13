import os as os
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

def write_output(label1, label2, label3, submission_file):
    """write the three label lists in order in submission_file"""
    with open(submission_file, 'w') as f:
        f.write('Id,Bound'+ '\n')
        for index, lab in enumerate(label1):
            f.write(str(index) + ',' + str(int(lab)) + '\n')
        for index, lab in enumerate(label2):
            f.write(str(len(label1) + index) + ',' + str(int(lab)) + '\n')
        for index, lab in enumerate(label3):
            f.write(str(len(label1) + len(label2) + index) + ',' + str(int(lab)))
            if index < len(label3) - 1:
                f.write('\n')
    
#folder_name = 'data'
#file_list = os.listdir(folder_name)

#for file in file_list:
#    matrix = read_file(os.path.join(folder_name, file))

# file_feat = 'Xtr1_mat100.csv'
# feat = read_file(os.path.join(folder_name, file))
# file_label = 'Ytr1.csv'
# label = read_file(os.path.join(folder_name, file))

#write_output(label, label, label, "submission_file.csv")
