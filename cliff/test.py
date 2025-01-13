import gymnasium as gym
import random
import numpy as np

def sample_action(action_values, eps):
    p = random.random()
    if p < eps:
        return random.randint(len(action_values))
    return action_values.argmax()

def make_q_table():
    return np.zeros((4, 12, 4))

def Q_agent(step_limit, *, initial_eps=1.0, alpha=0.1):
    env = gym.make('CliffWalking-v0')
    obs, info = env.reset()
    q_table = make_q_table()

env = gym.make('CliffWalking-v0', render_mode='human')
env.reset()
env.render()
