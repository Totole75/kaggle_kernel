import numpy as np
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


data_array = np.array([10,12,-5,8,-6,-4]).reshape((6, 1))
k = linear_kernel(data_array)
S = SVM([0,0,1,0,1,1], 0.0001, k)
alpha = S.optimize_alpha()
print("alpha", alpha)
print(S.test(alpha, data_array, [15,-1], np.array([0,1])))

