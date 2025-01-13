import numpy as np


class GreedyAgent:
    def __init__(self, k):
        self.k = k

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


class ConstantStepSizeAgent:
    def __init__(self, k, epsilon, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha

    def initialize(self):
        self.q = np.zeros(self.k)

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q)

    def update(self, a, r):
        self.q[a] += self.alpha * (r - self.q[a])

class UCBAgent:
    def __init__(self, k, c):
        self.k = k
        self.c = c

    def initialize(self):
        self.q = np.zeros(self.k)
        self.n = np.zeros(self.k)
        self.t = 0

    def act(self):
        self.t += 1
        ucb = self.q + self.c * np.sqrt(np.log(self.t) / (self.n + 1e-5))
        return np.argmax(ucb)

    def update(self, a, r):
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / self.n[a]

class GradientBanditAgent:
    def __init__(self, k, alpha):
        self.k = k
        self.alpha = alpha

    def initialize(self):
        self.h = np.zeros(self.k)
        self.pi = np.exp(self.h) / np.sum(np.exp(self.h))

    def act(self):
        return np.random.choice(self.k, p=self.pi)

    def update(self, a, r):
        self.h[a] += self.alpha * (r - np.mean(self.h))
        self.pi = np.exp(self.h) / np.sum(np.exp(self.h))
