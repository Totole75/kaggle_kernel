import numpy as np
from tools import *
from kernel import *
from algorithm import *

def grid_SVM(sigmas, lambdas, data_train, label_train, data_test, label_test):
    results = []
    for s in sigmas:
        for l in lambdas:
            kernel = levenshtein_kernel_from_dist(data_train, s)
            svm = SVM(label_train, l, kernel)
            alpha = svm.optimize_alpha()
            
            predictions = kernel.predict_from_dist(data_test, alpha)
            test_classes = []
            for i in range(len(label_test)):
                test_classes.append(0 if predictions[0, i] < 0 else 1)
            result = {"sigma": s,
                      "lambda": l,
                      "percentage_ones": np.mean(test_classes),
                      "success_rate": np.mean(np.equal(test_classes, ((label_test+1)/2).astype(int)))}
            print(result)
            results.append(result)
    return results

def cross_validate_SVM(nb_folds, low_sigma, high_sigma, nb_points_sigma, low_lambda, high_lambda, nb_points_lambda, data, label):
    nb_data = label.shape[0]
    int_array = np.array([i for i in range(nb_data)])
    np.random.shuffle(int_array)
    folds_idx = []
    nb_idx_fold = round(nb_data / nb_folds)
    for i in range(nb_folds):
        limit = (i + 1) * nb_idx_fold if i < nb_folds - 1 else nb_data
        sub_array = int_array[i * nb_idx_fold:limit]
        idx = np.array([True if i in sub_array else False for i in range(nb_data)])
        folds_idx.append(idx)
    all_results = []
    sigmas = [low_sigma * np.power(np.exp((1. / max(nb_points_sigma - 1, 1)) * np.log(high_sigma / low_sigma)), i) for i in range(nb_points_sigma)]
    lambdas = [low_lambda * np.power(np.exp((1. / max(nb_points_lambda - 1, 1)) * np.log(high_lambda / low_lambda)), i) for i in range(nb_points_lambda)]
    for idx in folds_idx:
        invert_idx = np.invert(idx)
        data_test = data[invert_idx, :]
        data_test = data_test[:, idx]
        label_test = label[idx]
        data_train = data[invert_idx, :]
        data_train = data_train[:, invert_idx]
        label_train = label[invert_idx]
        result = grid_SVM(sigmas, lambdas,
                          data_train, label_train, data_test, label_test)
        all_results += result
    mean_results = []
    for s in sigmas:
        for l in lambdas:
            mean_percentage_ones = np.mean([r["percentage_ones"] for r in all_results if s == r["sigma"] and l == r["lambda"]])
            mean_success_rate = np.mean([r["success_rate"] for r in all_results if s == r["sigma"] and l == r["lambda"]])
            result = {"sigma": s,
                      "lambda": l,
                      "percentage_ones": mean_percentage_ones,
                      "success_rate": mean_success_rate}
            mean_results.append(result)
    return mean_results
    

file_idx = 0
folder_name = "data/"

name_file_training = 'substring_test' + str(file_idx)
data_array = np.load(name_file_training + '.npy')[:500, :500]

result_file_name = "Ytr"+ str(file_idx) +".csv"
result_array = read_file(os.path.join(folder_name, result_file_name))[:500]

results = cross_validate_SVM(2, 1e-5, 1, 3, 1e-10, 3, 5, data_array, result_array)
print(max(results, key=lambda x: x["success_rate"]))


