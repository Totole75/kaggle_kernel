import numpy as np
from numba import jit, prange
from tqdm import tqdm

def transform_array(data_array):
    res_array = np.zeros((len(data_array), len(data_array[0])))
    for str_idx in range(len(data_array)):
        for idx in range(len(data_array[0])):
            res_array[str_idx, idx] = ord(data_array[str_idx][idx])
    return res_array

class abstract_kernel:
    """
    General class for creating kernels, used with heritage
    """
    def __init__(self, data_array, training_nb, center_kernel):
        self.data_array = data_array
        self.n = data_array.shape[0]
        self.training_nb = training_nb
        self.kernel_array = np.zeros((self.n, self.n))
        return

    def center_kernel_array(self):
        U_array = np.ones(self.kernel_array.shape) / float(self.n)
        centering_array = np.identity(self.n) - U_array
        self.kernel_array = (centering_array).dot(self.kernel_array.dot(centering_array))

    def normalize_kernel(self, normalizing_coef):
        self.kernel_array = self.kernel_array / float(normalizing_coef)

    def predict(self, test_array, alpha):
        K = self.kernel_array[:self.training_nb, :][:, self.training_nb:]
        #K = self.compute_kernel(self.data_array, test_array)
        return(alpha.reshape(1,-1).dot(K))

    def compute_kernel(self, first_array, second_array):
        raise NotImplementedError()

class imported_kernel(abstract_kernel):
    """
    Kernel whose array is imported from a .npy file
    """
    def __init__(self, data_array, training_nb, load_path, center_kernel=True):
        abstract_kernel.__init__(self, data_array, training_nb, center_kernel)
        self.kernel_array = np.load(load_path)
        if center_kernel:
            abstract_kernel.center_kernel_array(self)

    def __init__(self, data_array, training_nb, center_kernel=True):
        abstract_kernel.__init__(self, data_array, training_nb, center_kernel)
        self.kernel_array = data_array
        if center_kernel:
            abstract_kernel.center_kernel_array(self)


class linear_kernel(abstract_kernel):
    """
    Linear kernel : just your regular scalar product
    """
    def __init__(self, data_array, training_nb, center_kernel=True):
        abstract_kernel.__init__(self, data_array, training_nb, center_kernel)
        # Computing the kernel array
        self.kernel_array = np.dot(data_array, data_array.T)
        if center_kernel:
            abstract_kernel.center_kernel_array(self)

    def compute_kernel(self, first_array, second_array):
        return np.dot(first_array, second_array.T)

class gaussian_kernel(abstract_kernel):
    def __init__(self, data_array, training_nb, sigma, center_kernel=True):
        abstract_kernel.__init__(self, data_array, training_nb, center_kernel)
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
    def __init__(self, data_array, training_nb, alpha, constant=0, center_kernel=True):
        abstract_kernel.__init__(self, data_array, training_nb, center_kernel)
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
    def __init__(self, data_array, training_nb, sigma, center_kernel=True):
        abstract_kernel.__init__(self, data_array, training_nb, center_kernel)
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

class substring_kernel(abstract_kernel):
    """
    Substring kernel
    """
    def __init__(self, data_array, training_nb, k, lmb, center_kernel=True):
        abstract_kernel.__init__(self, data_array, training_nb, center_kernel)
        self.k = k
        self.lmb = lmb
        # Computing the kernel array
        self.kernel_array = self.compute_kernel(data_array, data_array)  
        if center_kernel:
            abstract_kernel.center_kernel_array(self)

    def compute_kernel(self, first_array, second_array):
        @jit(nopython=True, fastmath=True)
        def compute_rec_val(str_a, str_b, len_a, len_b, k, lmb, rec_val):
            for cur_k in range(1, k+1):
                for cur_i in range(cur_k, len_a+1):
                    for cur_j in range(cur_k, len_b+1):
                        rec_val[cur_k, cur_i, cur_j] = lmb*(rec_val[cur_k, cur_i-1, cur_j] + rec_val[cur_k, cur_i, cur_j-1] - lmb*rec_val[cur_k, cur_i-1, cur_j-1])
                        if (str_a[cur_i-1] == str_b[cur_j-1]):
                            rec_val[cur_k, cur_i, cur_j] += lmb*lmb*rec_val[cur_k-1, cur_i-1, cur_j-1]
            return rec_val

        @jit(nopython=True, fastmath=True)
        def compute_kernel_cell(str_a, str_b, k, lmb):
            len_a = len(str_a)
            len_b = len(str_b)
            
            # Computing the recursive side function B
            rec_val = np.zeros((k+1, len_a+1, len_b+1))
            rec_val[0, :, :] = np.ones((len_a+1, len_b+1))
            # Completing the array of the recursion function
            rec_val = compute_rec_val(str_a, str_b, len_a, len_b, k, lmb, rec_val)
            
            # Computing the final value
            ker_val = np.zeros((len_a+1, len_b+1))
            # Finishing
            for cur_i in range(k-1, len_a):
                sec_term = 0
                for cur_j in range(1, k+1):
                    if (str_b[cur_j-1] == str_a[cur_i]):
                        sec_term += rec_val[k-1, cur_i, cur_j-1]
                ker_val[cur_i+1, k] = ker_val[cur_i, k] + lmb*lmb*sec_term
            for cur_j in range(k, len_b):
                sec_term = 0
                for cur_i in range(1, len_a+1):
                    if (str_a[cur_i-1] == str_b[cur_j]):
                        sec_term += rec_val[k-1, cur_i-1, cur_j]
                ker_val[len_a, cur_j+1] = ker_val[len_a, cur_j] + lmb*lmb*sec_term
                
            return ker_val[len_a, len_b]

        @jit(nopython=True, parallel=True)
        def jit_compute_kernel(first_array, second_array, equal_arrays, k, lmb):
            kernel_array = np.zeros((first_array.shape[0], second_array.shape[0]))
            if equal_arrays:
                for i in range(first_array.shape[0]):
                    for j in range(i, second_array.shape[0]):
                        kernel_value = compute_kernel_cell(first_array[i], second_array[j], k, lmb)
                        kernel_array[i,j] = kernel_value
                        kernel_array[j,i] = kernel_value
            else:
                for i in prange(first_array.shape[0]):
                    for j in range(second_array.shape[0]):
                        kernel_value = compute_kernel_cell(first_array[i], second_array[j], k, lmb)
                        kernel_array[i,j] = kernel_value
            return kernel_array
        
        first_array = transform_array(first_array)
        second_array = transform_array(second_array)
        # Calls it on a dummy to compile, takes around 3 seconds
        jit_compute_kernel(first_array[:2], second_array[:2], np.array_equal(first_array, second_array), self.k, self.lmb)
        # Computing the REAL kernel array
        result_array = jit_compute_kernel(first_array, second_array, np.array_equal(first_array, second_array), self.k, self.lmb)   
        
        return result_array
