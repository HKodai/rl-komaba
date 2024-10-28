import numpy as np


def experiment(testbed, agents, t, n):
    average_reward = []
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
        optimal_action.append(np.mean(all_optimal, axis=0))
    return average_reward, optimal_action
