import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
import cplex
from kernel import *

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
        #print("bound", bound)
        c = np.array(self.labels_pos_neg)
        H = np.array(self.kernel_array)
        A = np.concatenate([np.diag(self.labels_pos_neg), - np.diag(self.labels_pos_neg)], axis=0)
        b = np.concatenate([np.array(bound * np.ones(self.n)), np.zeros(self.n)], axis=0)
        solver = cplex.Cplex()
        solver.variables.add(names = [str(i) for i in range(self.n)],
                             lb = [0.0 if self.labels_pos_neg[i] == 1 else - bound for i in range(self.n)],
                             ub = [0.0 if self.labels_pos_neg[i] == - 1 else bound for i in range(self.n)],
                             types = [solver.variables.type.continuous] * self.n)

        list_coeffs_lin = [(str(i), 2 * float(c[i])) for i in range(self.n)]
        solver.objective.set_linear(list_coeffs_lin)

        list_coeffs_quad = []
        for i in range(self.n):
            for j in range(i, self.n):
                list_coeffs_quad.append((i, j, - 2 * H[i][j] if i == j else - H[i][j]))
        solver.objective.set_quadratic_coefficients(list_coeffs_quad)

        solver.objective.set_sense(solver.objective.sense.maximize)

        """solver.linear_constraints.add(lin_expr = [cplex.SparsePair(ind=[str(i)], val=[float(self.labels_pos_neg[i])]) for i in range(self.n)],
                                      senses = ["L"] * self.n,
                                      rhs = [bound] * self.n,
                                      names = ["c" + str(i) for i in range(self.n)])
        solver.linear_constraints.add(lin_expr = [cplex.SparsePair(ind=[str(i)], val=[float(self.labels_pos_neg[i])]) for i in range(self.n)],
                                      senses = ["G"] * self.n,
                                      rhs = [0.0] * self.n,
                                      names = ["c" + str(i + self.n) for i in range(self.n)])"""
                                      
        solver.parameters.optimalitytarget.set(3)
        solver.set_log_stream(None)
        solver.set_error_stream(None)
        solver.set_warning_stream(None)
        solver.set_results_stream(None)
        solver.solve()
        status = solver.solution.get_status()
        print(solver.solution.status[status])
        #solver.write("test.lp")
        #print(solver.objective.get_num_quadratic_nonzeros())
        #print(solver.solution.get_objective_value())
        #print(solver.solution.get_values([str(i) for i in range(self.n)]))
        return np.array(solver.solution.get_values([str(i) for i in range(self.n)]))

def ridge_regression(K, label_array, lambda_reg):
    n = K.shape[0]
    alpha = np.linalg.inv(K + lambda_reg*n*np.eye(n))
    alpha = alpha.dot(label_array)
    return(alpha)

class Kmeans:

    def __init__(self, kernel):
        self.kernel = kernel
        self.kernel_array = kernel.kernel_array
        self.n = self.kernel_array.shape[0]

    def create_clusters(self, cluster_nb, restart_nb=3):
        self.cluster_nb = cluster_nb
        all_clusters = []
        all_energies = []
        for restart_idx in tqdm(range(restart_nb)):
            # Initialization : creating random clusters
            clusters = []
            for idx in range(cluster_nb):
                clusters.append([])
            cluster_rep = np.random.randint(cluster_nb, size=self.n)
            for idx in range(cluster_nb):
                #  Done in order to avoid empty clusters at the beginning
                clusters[idx].append(idx)
            for idx in range(cluster_nb, self.n):
                clusters[cluster_rep[idx]].append(idx)
            # Turning them into arrays for better handling
            for idx in range(cluster_nb):
                clusters[idx] = np.array(clusters[idx])
            # Optimization part
            first_iter = True
            diff_clusters = True
            old_clusters = [np.array([]) for _ in range(cluster_nb)]
            while (first_iter or diff_clusters):
                first_iter = False
                first_term = np.zeros(cluster_nb)
                third_term = np.zeros((cluster_nb, self.n))
                for idx in range(cluster_nb):
                    first_term[idx] = self.kernel_array[clusters[idx],:][:,clusters[idx]].mean() 
                    third_term[idx, :] = self.kernel_array[clusters[idx], :].mean(axis=0)
                first_term = np.tile(first_term, (self.n, 1)).T
                second_term = np.diag(self.kernel_array)
                second_term = np.tile(second_term, (cluster_nb, 1))
                squared_distances = first_term + second_term - 2*third_term
                cluster_rep = np.argmin(squared_distances, axis=0)
                # Forming the new clusters
                clusters = []
                for idx in range(cluster_nb):
                    clusters.append([])
                for idx in range(self.n):
                    clusters[cluster_rep[idx]].append(idx)
                # Deleting the empty clusters
                clusters_to_delete = []
                for idx in range(cluster_nb):
                    if (len(clusters[idx]) != 0):
                        clusters[idx] = np.array(clusters[idx])
                    else:
                        clusters_to_delete.append(idx)
                for idx in reversed(clusters_to_delete):
                    del clusters[idx]
                cluster_nb = len(clusters)

                diff_clusters = False
                for idx in range(cluster_nb):
                    # Checking if the clusters have changed since last iteration
                    if (not np.array_equal(old_clusters[idx], clusters[idx])):
                        diff_clusters = True
                    old_clusters[idx] = clusters[idx]

            energy = 0
            for cluster_idx in range(cluster_nb):
                if (clusters[cluster_idx].shape[0] != 0):
                    cluster_energy = 0
                    for idx_i in clusters[cluster_idx]:
                        for idx_j in clusters[cluster_idx]:
                            cluster_energy += self.kernel_array[idx_i, idx_j]
                    cluster_energy = cluster_energy / float(clusters[cluster_idx].shape[0])
                    energy += cluster_energy
            all_energies.append(energy)
            all_clusters.append(clusters)

        # Keeping the clustering that maximizes the objective
        max_energy_idx = np.argmax(np.array(all_energies))
        self.clusters = all_clusters[max_energy_idx]

    def predict(self, training_results, test_array):
        test_nb = test_array.shape[0]
        # Computing the majority class of each cluster
        cluster_classes = np.zeros(self.cluster_nb)
        for idx in range(self.cluster_nb):
            cluster_classes[idx] = round(training_results[self.clusters[idx]].mean())
        # Computing distances between the test values and the cluster centroids
        first_term = np.zeros(self.cluster_nb)
        third_term = np.zeros((self.cluster_nb, self.n))
        for idx in range(self.cluster_nb):
            first_term[idx] = self.kernel_array[self.clusters[idx],:][:,self.clusters[idx]].mean() 
            third_term[idx, :] = self.kernel_array[self.clusters[idx], :].mean(axis=0)
        first_term = np.tile(first_term, (self.n, 1)).T
        second_term = np.diag(self.kernel_array)
        second_term = np.tile(second_term, (self.cluster_nb, 1))
        
        squared_distances = first_term + second_term - 2*third_term
        # Assigning each test value to a cluster
        cluster_test_rep = np.argmin(squared_distances, axis=0)
        # Assigning the cluster class to the test value
        test_classes = np.zeros(test_nb)
        for idx in range(test_nb):
            test_classes[idx] = cluster_classes[cluster_test_rep[idx]]

        return test_classes
