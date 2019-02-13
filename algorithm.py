import numpy as np

def ridge_regression(K, label_array, lambda_reg):
    n = K.shape[0]
    alpha = np.linalg.inv(K + lambda_reg*n*np.eye(n))
    alpha = alpha.dot(label_array)
    return(alpha)
    