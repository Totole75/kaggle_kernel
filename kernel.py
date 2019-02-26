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

class sigmoid_kernel(abstract_kernel):
    def __init__(self, data_array, alpha, constant=0, center_kernel=True):
        abstract_kernel.__init__(self, data_array, center_kernel)
        self.alpha = alpha
        self.constant = constant
        self.kernel_array = self.compute_kernel(data_array, data_array)
        if center_kernel:
            abstract_kernel.center_kernel_array(self)
        
    def compute_kernel(self, data_array_1, data_array_2):
        multiplied_matrix = np.dot(data_array_1, data_array_2.T)
        kernel_array = np.tanh(self.alpha * multiplied_matrix + self.constant)
        return(kernel_array)
    
class levenshtein_kernel_from_dist(abstract_kernel):
    def __init__(self, dist_matrix, sigma, center_kernel=True):
        abstract_kernel.__init__(self, dist_matrix, center_kernel)
        self.sigma = sigma
        self.kernel_array = self.compute_kernel_dist(dist_matrix)
        if center_kernel:
            abstract_kernel.center_kernel_array(self)
            
    def compute_kernel_dist(self, dist_matrix):
        k = dist_matrix*(- 1./(2 * np.power(self.sigma, 2)))
        kernel_array = np.exp(k)
        return(kernel_array)
        
    def predict_from_dist(self, dist_matrix, alpha):
        K = self.compute_kernel_dist(dist_matrix)
        return(alpha.reshape(1,-1).dot(K))
    
class levenshtein_kernel(abstract_kernel):
    def __init__(self, data_array, sigma, center_kernel=True):
        abstract_kernel.__init__(self, data_array, center_kernel)
        self.sigma = sigma
        self.kernel_array = self.compute_kernel(data_array)
        if center_kernel:
            abstract_kernel.center_kernel_array(self)
            
    def compute_kernel(self, data_array_1, data_array_2=[[-1]]):
        if data_array_2[0][0] != -1:
            #distance avec le vecteur de test
            n1 = data_array_1.shape[0]
            n2 = data_array_2.shape[0]
            k = np.zeros((n1, n2))
        
            for i in tqdm(range(n1)):
                for j in range(n2):
                    d = distance(str(data_array_1[i,:]),str(data_array_2[j,:]))
                    k[i,j] = d
                    
        if data_array_2[0][0] == -1:
            # distance avec elle meme
            n1 = data_array_1.shape[0]
            n2 = data_array_1.shape[0]
            k = np.zeros((n1, n2))
        
            for i in tqdm(range(n1)):
                for j in range(i,n2):
                    d = distance(str(data_array_1[i,:]),str(data_array_1[j,:]))
                    k[i,j] = d
                    k[j,i] = d
        k *= (- 1./(2 * np.power(self.sigma, 2)))
        kernel_array = np.exp(k)
        return(kernel_array)
