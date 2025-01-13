import numpy as np
from itertools import accumulate

def experiment(testbed, agents, t, n):
    average_reward = []
    cumulative_reward = []
    optimal_action = []
    for agent in agents:
        all_rewards = []
        all_optimal = []
        for _ in range(n):
            testbed.initialize()
            agent.initialize()
            rewards = []
            optimal = []
            for _ in range(t):
                a = agent.act()
                r = testbed.get_r(a)
                agent.update(a, r)
                testbed.move()
                rewards.append(r)
                optimal.append(a == testbed.optimal)
            all_rewards.append(np.array(rewards))
            all_optimal.append(np.array(optimal))
        average_reward.append(np.mean(all_rewards, axis=0))
        cumulative_reward.append(list(accumulate(np.mean(all_rewards, axis=0))))
        optimal_action.append(np.mean(all_optimal, axis=0))
    return average_reward, cumulative_reward, optimal_action

def parameter_study(testbed, agents, t, n):
    result = [[0 for _ in range(len(agents[i]))] for i in range(len(agents))]
    for _ in range(n):
        testbed.initialize()
        for i in range(t):
            for j in range(len(agents)):
                for k in range(len(agents[j])):
                    agent = agents[j][k]
                    agent.initialize()
                    a = agent.act()
                    r = testbed.get_r(a)
                    agent.update(a, r)
                    if i > t//2:
                        result[j][k] += r
            testbed.move()
    for i in range(len(agents)):
        for j in range(len(agents[i])):
            result[i][j] /= (n * t // 2)
    return result