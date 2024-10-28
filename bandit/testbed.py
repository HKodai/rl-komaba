import numpy as np


class StationaryTestbed:
    def __init__(self, k, mean, var1, var2):
        self.k = k
        self.mean = mean
        self.var1 = var1
        self.var2 = var2

    def initialize(self):
        self.q = np.random.normal(self.mean, self.var1, self.k)
        self.optimal = np.argmax(self.q)

    def move(self):
        pass

    def get_r(self, a):
        return np.random.normal(self.q[a], self.var2)


class NonStationaryTestbed:
    def __init__(self, k, var, std):
        self.k = k
        self.var = var
        self.std = std

    def initialize(self):
        self.q = np.zeros(self.k)
        self.optimal = np.random.randint(self.k)

    def move(self):
        self.q += np.random.normal(0, self.std, self.k)
        self.optimal = np.argmax(self.q)

    def get_r(self, a):
        return np.random.normal(self.q[a], self.var)
