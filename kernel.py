import numpy as np

class linear_kernel:
    """
    Linear kernel : just your regular scalar product
    """
    def __init__(self, data_array):
        self.data_array = data_array
        # Computing the kernel array
        self.kernel_array = np.dot(data_array, data_array.T)