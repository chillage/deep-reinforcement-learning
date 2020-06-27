import numpy as np
import math
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.episode_num = 1

        self.min_epsilon = 0.005
        self.max_epsilon = 0.7
        self.min_epsilon_iter_val = 15000
        self.alpha = 0.01
        self.epsilon = self.get_epsilon()

    def get_epsilon(self):
        # return 0.9

        if self.episode_num >= self.min_epsilon_iter_val:
            return self.min_epsilon

        return self.max_epsilon + (self.min_epsilon - self.max_epsilon) * self.episode_num / self.min_epsilon_iter_val

        #return math.pow(2, math.log2(self.max_epsilon) + (math.log2(self.min_epsilon) - math.log2(self.max_epsilon)) * self.episode_num/ (self.min_epsilon_iter_val))

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
        if done:
            self.episode_num += 1
            self.epsilon = self.get_epsilon()

        expected_next_action_distribution = self.epsilon_greedy_distribution(next_state)
        expected_next_action_value = sum([self.Q[next_state][a] * a_prob for a, a_prob in enumerate(expected_next_action_distribution)])
        self.Q[state][action] = reward + expected_next_action_value  # alpha == 1