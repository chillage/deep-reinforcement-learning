import numpy as np
import math
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, epsilon=0.08926, gamma=0.8597, epsilon_divisor = 17.87):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_divisor = epsilon_divisor

    def epsilon_greedy_distribution(self, state):
        dist: np.ndarray = np.zeros(self.nA) + self.epsilon / self.nA
        maxAction = self.Q[state].argmax()
        dist[maxAction] += 1 - self.epsilon
        return dist

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.random.choice(self.nA, p=self.epsilon_greedy_distribution(state))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        expected_next_action_distribution = self.epsilon_greedy_distribution(next_state)
        expected_next_action_value = sum([self.Q[next_state][a] * a_prob for a, a_prob in enumerate(expected_next_action_distribution)])
        self.Q[state][action] = reward + self.gamma * expected_next_action_value  # alpha == 1

        if done:
            self.epsilon /= self.epsilon_divisor