import numpy as np

class GreedyAgent:
    def __init__(self, k):
        self.k = k
        self.q = np.zeros(k)
        self.n = np.zeros(k)
    def initialize(self):
        self.q = np.zeros(self.k)
        self.n = np.zeros(self.k)
    def act(self):
        return np.argmax(self.q)
    def update(self, a, r):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / self.n[a]

class EpsilonGreedyAgent:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.q = np.zeros(k)
        self.n = np.zeros(k)
    def initialize(self):
        self.q = np.zeros(self.k)
        self.n = np.zeros(self.k)
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q)
    def update(self, a, r):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / self.n[a]
