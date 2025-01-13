import matplotlib.pyplot as plt
from testbed import NonStationaryTestbed
from agent import EpsilonGreedyAgent, ConstantStepSizeAgent, UCBAgent, GradientBanditAgent, OptimisticGreedyAgent
from experiment import parameter_study

k = 10
mean = 0
var1 = 1
var2 = 1
t = 100
n = 2000
testbed = NonStationaryTestbed(k, var1, 0.01)
agents = [[EpsilonGreedyAgent(k, 1/128), EpsilonGreedyAgent(k, 1/64), EpsilonGreedyAgent(k, 1/32), EpsilonGreedyAgent(k, 1/16), EpsilonGreedyAgent(k, 1/8), EpsilonGreedyAgent(k, 1/4)],
          [ConstantStepSizeAgent(k, 1/128), ConstantStepSizeAgent(k, 1/64), ConstantStepSizeAgent(k, 0.1, 1/32), ConstantStepSizeAgent(k, 0.1, 1/16), ConstantStepSizeAgent(k, 0.1, 1/8), ConstantStepSizeAgent(k, 0.1, 1/4)],
          [UCBAgent(k, 1/16), UCBAgent(k, 1/8), UCBAgent(k, 1/4), UCBAgent(k, 1/2), UCBAgent(k, 1), UCBAgent(k, 2), UCBAgent(k, 4)],
          [GradientBanditAgent(k, 1/32), GradientBanditAgent(k, 1/16), GradientBanditAgent(k, 1/8), GradientBanditAgent(k, 1/4), GradientBanditAgent(k, 1/2), GradientBanditAgent(k, 1), GradientBanditAgent(k, 2)]]
result = parameter_study(testbed, agents, t, n)
print(result)
