import numpy as np

class linear_kernel:
    """
    Linear kernel : just your regular scalar product
    """
    def __init__(self, data_array):
        self.data_array = data_array
        # Computing the kernel array
        self.kernel_array = np.dot(data_array, data_array.T)
    
class gaussian_kernel:
    def __init__(self, data_array, sigma):
        self.data_array = data_array
        self.sigma = sigma
        self.n = self.data_array.shape[0]
        self.kernel_array = self.compute_kernel(data_array, data_array)
        
    def compute_kernel(self, data_array_1, data_array_2):
        trnorms1 = np.mat([(np.dot(v,v.T)) for v in data_array_1]).T
        trnorms2 = np.mat([(np.dot(v,v.T)) for v in data_array_2]).T
        k1 = trnorms1 * np.mat(np.ones((data_array_2.shape[0], 1), dtype=np.float64)).T
        k2 = np.mat(np.ones((self.n, 1), dtype=np.float64)) * trnorms2.T
        k = k1 + k2 - 2* np.dot(data_array_1, data_array_2.T)
        k *= - 1./(2 * np.power(self.sigma, 2))
        kernel_array = np.exp(k)
        return(kernel_array)
        
    def predict(self, test_array, alpha):
        K = self.compute_kernel(self.data_array, test_array)
        return(alpha.reshape(1,-1).dot(K))
        