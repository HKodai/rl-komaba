import numpy as np

class StationaryProblem:
    def __init__(self, k, mean, var1, var2):
        self.k = k
        self.q = np.random.normal(mean, var1, k)
        self.optimal = np.argmax(self.q)
        self.var2 = var2
    def get_r(self, a):
        return np.random.normal(self.q[a], self.var2)
    