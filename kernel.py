import numpy as np

class abstract_kernel:
    """
    General class for creating kernels, used with heritage
    """
    def __init__(self, data_array, center_kernel):
        self.data_array = data_array
        self.n = data_array.shape[0]
        self.kernel_array = np.zeros((self.n, self.n))
        return

    def center_kernel_array(self):
        U_array = np.ones(self.kernel_array.shape) / float(self.n)
        centering_array = np.identity(self.n) - U_array
        self.kernel_array = (centering_array).dot(self.kernel_array.dot(centering_array))

    def predict(self, test_array, alpha):
        K = self.compute_kernel(self.data_array, test_array)
        return(alpha.reshape(1,-1).dot(K))

    def compute_kernel(self, first_array, second_array):
        raise NotImplementedError()


class linear_kernel(abstract_kernel):
    """
    Linear kernel : just your regular scalar product
    """
    def __init__(self, data_array, center_kernel=True):
        abstract_kernel.__init__(self, data_array, center_kernel)
        # Computing the kernel array
        self.kernel_array = np.dot(data_array, data_array.T)
        if center_kernel:
            abstract_kernel.center_kernel_array(self)

    def compute_kernel(self, first_array, second_array):
        return np.dot(first_array, second_array.T)

class gaussian_kernel(abstract_kernel):
    def __init__(self, data_array, sigma, center_kernel=True):
        abstract_kernel.__init__(self, data_array, center_kernel)
        self.sigma = sigma
        self.kernel_array = self.compute_kernel(data_array, data_array)
        if center_kernel:
            abstract_kernel.center_kernel_array(self)
        
    def compute_kernel(self, data_array_1, data_array_2):
        trnorms1 = np.mat([(np.dot(v,v.T)) for v in data_array_1]).T
        trnorms2 = np.mat([(np.dot(v,v.T)) for v in data_array_2]).T
        k1 = trnorms1 * np.mat(np.ones((data_array_2.shape[0], 1), dtype=np.float64)).T
        k2 = np.mat(np.ones((data_array_1.shape[0], 1), dtype=np.float64)) * trnorms2.T
        k = k1 + k2 - 2* np.dot(data_array_1, data_array_2.T)
        k *= - 1./(2 * np.power(self.sigma, 2))
        kernel_array = np.exp(k)
        return(kernel_array)
