import matplotlib.pyplot as plt
from testbed import NonStationaryTestbed
from agent import EpsilonGreedyAgent, ConstantStepSizeAgent
from experiment import experiment

k = 10
mean = 0
var1 = 1
var2 = 1
t = 10000
n = 2000
testbed = NonStationaryTestbed(k, var1, 0.01)
agents = [EpsilonGreedyAgent(k, 0.1), ConstantStepSizeAgent(k, 0.1, 0.1)]
reward, optimal = experiment(testbed, agents, t, n)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(reward[0], label="sample-average", color="blue")
ax1.plot(reward[1], label="constant alpha", color="red")
ax1.legend()
ax1.set_ylabel("Average reward")
ax1.set_xlabel("Steps")
ax2 = fig.add_subplot(212)
ax2.plot(optimal[0], label="sample-average", color="blue")
ax2.plot(optimal[1], label="constant alpha", color="red")
ax2.legend()
ax2.set_ylabel("Optimal action")
ax2.set_xlabel("Steps")
plt.savefig("nonstationary.png")
