import matplotlib.pyplot as plt
from testbed import StationaryProblem
from agent import GreedyAgent, EpsilonGreedyAgent
from experiment import experiment

k = 10
mean = 0
var1 = 1
var2 = 1
t = 1000
n = 2000
problem = StationaryProblem(k, mean, var1, var2)
agents = [GreedyAgent(k), EpsilonGreedyAgent(k, 0.1), EpsilonGreedyAgent(k, 0.01)]
result = experiment(problem, agents, t, n)
plt.plot(result[0], label="Greedy")
plt.plot(result[1], label="Epsilon=0.1")
plt.plot(result[2], label="Epsilon=0.01")
plt.legend()
plt.show()
