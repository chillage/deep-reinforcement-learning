import gym
import random
import torch
import numpy as np
from collections import deque
from dqn_agent import Agent

const_seed = 0

env = gym.make('LunarLander-v2')
env.seed(const_seed)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=const_seed)


