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
        n = self.data_array.shape[0]
        trnorms1 = np.mat([(np.dot(v,v.T)) for v in self.data_array]).T
        k1 = trnorms1 * np.mat(np.ones((n, 1), dtype=np.float64)).T
        k2 = np.mat(np.ones((n, 1), dtype=np.float64)) * trnorms1.T
        k = k1 + k2 - 2* np.dot(self.data_array, self.data_array.T)
        k *= - 1./(2 * np.power(sigma, 2))
        
        self.kernel_array = np.exp(k)