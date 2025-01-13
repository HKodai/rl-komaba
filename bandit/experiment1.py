import matplotlib.pyplot as plt
from testbed import StationaryTestbed
from agent import GreedyAgent, EpsilonGreedyAgent
from experiment import experiment

k = 10
mean = 0
var1 = 1
var2 = 1
t = 4000
n = 2000
testbed = StationaryTestbed(k, mean, var1, var2)
agents = [GreedyAgent(k), EpsilonGreedyAgent(k, 0.1), EpsilonGreedyAgent(k, 0.01)]
reward, cumulative, optimal = experiment(testbed, agents, t, n)
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(reward[0], label="Greedy", color="green")
ax1.plot(reward[1], label="Epsilon=0.1", color="blue")
ax1.plot(reward[2], label="Epsilon=0.01", color="red")
ax1.legend()
ax1.set_ylabel("Average reward")
ax1.set_xlabel("Steps")
ax2 = fig.add_subplot(312)
ax2.plot(cumulative[0], label="Greedy", color="green")
ax2.plot(cumulative[1], label="Epsilon=0.1", color="blue")
ax2.plot(cumulative[2], label="Epsilon=0.01", color="red")
ax2.legend()
ax2.set_ylabel("Cumulative reward")
ax2.set_xlabel("Steps")
ax2 = fig.add_subplot(313)
ax2.plot(optimal[0], label="Greedy", color="green")
ax2.plot(optimal[1], label="Epsilon=0.1", color="blue")
ax2.plot(optimal[2], label="Epsilon=0.01", color="red")
ax2.legend()
ax2.set_ylabel("Optimal action")
ax2.set_xlabel("Steps")
plt.show()
