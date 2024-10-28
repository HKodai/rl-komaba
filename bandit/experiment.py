import numpy as np

def experiment(problem, agents, t, n):
    result = []
    for agent in agents:
        all_rewards = []
        for _ in range(n):
            agent.initialize()
            rewards = []
            for _ in range(t):
                a = agent.act()
                r = problem.get_r(a)
                agent.update(a, r)
                rewards.append(r)
            all_rewards.append(np.array(rewards))
        result.append(np.mean(all_rewards, axis=0))
    return result