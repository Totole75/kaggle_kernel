import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from kernel import linear_kernel

class SVM:

    def __init__(self, labels, lamb, kernel):
        self.labels = labels
        self.labels_pos_neg = np.array([1 if l == 1 else -1 for l in self.labels])
        self.lamb = lamb
        self.kernel = kernel
        self.kernel_array = kernel.kernel_array
        self.n = len(self.labels)

    def optimize_alpha(self):
        bound = (1 / (2 * self.lamb * self.n))
        print("bound", bound)
        c = np.array(self.labels_pos_neg)
        H = np.array(self.kernel_array)
        A = np.concatenate([np.diag(self.labels_pos_neg), - np.diag(self.labels_pos_neg)], axis=0)
        b = np.concatenate([np.array(bound * np.ones(self.n)), np.zeros(self.n)], axis=0)
        #print("A:", A)
        #print("H:", H)
        #print("b:", b)
        #print("c:", c)
        alpha_init = np.zeros(self.n).reshape((self.n, 1))

        def objective(x, sign=-1.):
            return sign * (- 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(c, x))

        def jacobien(x, sign=-1.):
            return sign * (np.dot(x.T, H) + c)

        constraints = {"type": "ineq",
                       "fun": lambda x: b - np.dot(A, x),
                       "jac": lambda x: - A}

        options = {"disp": False}

        solve = minimize(objective, alpha_init, jac=jacobien, constraints=constraints, method='SLSQP', options=options)

        return solve['x']

    def test(self, alpha, data_training, data_test, real_labels):
        predicted_labels = self.predict(alpha, data_training, data_test)
        print(predicted_labels)
        return sum(np.equal(predicted_labels, real_labels)) / len(real_labels)

    def predict(self, alpha, data_training, data_test):
        labels = []
        for data in data_test:
            k_values = np.array([self.kernel.compute_value(data, data_train) for data_train in data_training])
            f_predict = np.sum(alpha * k_values)
            if f_predict > 0:
                labels.append(1)
            else:
                labels.append(0)
            print(f_predict)
        return np.array(labels)

def ridge_regression(K, label_array, lambda_reg):
    n = K.shape[0]
    alpha = np.linalg.inv(K + lambda_reg*n*np.eye(n))
    alpha = alpha.dot(label_array)
    return(alpha)

def kernel_kmeans(kernel, cluster_nb, step_number=100):
    kernel_array = kernel.kernel_array
    # Initialization : creating random clusters
    clusters = []
    for idx in range(cluster_nb):
        clusters.append([])
    cluster_rep = np.random.randint(cluster_nb, size=kernel.n)
    for idx in range(cluster_nb):
        #  Done in order to avoid empty clusters at the beginning
        clusters[idx].append(idx)
    for idx in range(cluster_nb, kernel.n):
        clusters[cluster_rep[idx]].append(idx)
    # Turning them into arrays for better handling
    for idx in range(cluster_nb):
        clusters[idx] = np.array(clusters[idx])
    # Optimization part
    for step_idx in tqdm(range(step_number)):
        kernel_means = np.zeros((cluster_nb, kernel.n))
        for idx in range(cluster_nb):
            if (clusters[idx].shape[0] != 0):
                # And else we do nothing
                kernel_means[idx,:] = kernel_array[clusters[idx], :].mean(axis=0)
        cluster_rep = np.argmax(kernel_means, axis=0)
        # Forming the new clusters
        clusters = []
        for idx in range(cluster_nb):
            clusters.append([])
        for idx in range(kernel.n):
            clusters[cluster_rep[idx]].append(idx)
        for idx in range(cluster_nb):
            clusters[idx] = np.array(clusters[idx])
    return clusters

def cluster_test(clusters, kernel,
                training_results,
                test_array, test_results):
    # Deleting the empty clusters
    clusters_to_delete = []
    for idx in range(len(clusters)):
        if (clusters[idx].shape[0] == 0):
            clusters_to_delete.append(idx)
    for idx in reversed(clusters_to_delete):
        del clusters[idx]
    cluster_nb = len(clusters)
    test_nb = test_results.shape[0]
    kernel_array = kernel.kernel_array
    # Computing the majority class of each cluster
    cluster_classes = np.zeros(cluster_nb)
    for idx in range(cluster_nb):
        cluster_classes[idx] = round(training_results[clusters[idx]].mean())
    kernel_values = kernel.compute_kernel_against(test_array)
    # Computing distances between the test values and the cluster centroids
    first_term = np.zeros(cluster_nb)
    third_term = np.zeros((cluster_nb, test_nb))
    for idx in range(cluster_nb):
        first_term[idx] = kernel_array[clusters[idx], clusters[idx]].mean()
        third_term[idx, :] = kernel_values[clusters[idx], :].mean(axis=0)
    first_term = np.tile(first_term, (test_nb, 1)).T
    second_term = np.diag(kernel.compute_kernel(test_array, test_array))
    second_term = np.tile(second_term, (cluster_nb, 1))

    squared_distances = first_term + second_term - 2*third_term
    # Assigning each test value to a cluster
    cluster_test_rep = np.argmin(squared_distances, axis=0)
    # Assigning the cluster class to the test value
    test_classes = np.zeros(test_nb)
    for idx in range(test_nb):
        test_classes[idx] = cluster_classes[cluster_test_rep[idx]]

    return test_classes
