# custom utilies for displaying animation, collecting rollouts and more
import pong_utils
import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from parallelEnv import parallelEnv
import progressbar as pb

if __name__ == '__main__':
if __name__ == '__main__':

    # PongDeterministic does not contain random frameskip
    # so is faster to train than the vanilla Pong-v4 environment
    env = gym.make('PongDeterministic-v4')

    # check which device is being used.
    # I recommend disabling gpu until you've made sure that the code runs
    device = pong_utils.device

    policy=pong_utils.Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    def clipped_surrogate(policy, old_probs, states, actions, rewards,
                          discount=0.995, epsilon=0.1, beta=0.01):
        discount = discount ** np.arange(len(rewards))
        rewards = np.asarray(rewards) * discount[:, np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = pong_utils.states_to_prob(policy, states)
        new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0 - new_probs)

        ratio = new_probs / old_probs

        loss = torch.min(ratio, torch.clamp(ratio, 1-epsilon, 1+epsilon))

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) +
                    (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        return torch.mean(loss * rewards + beta * entropy)


    # training loop max iterations
    episode = 500

    # widget bar to display progress

    widget = ['training loop: ', pb.Percentage(), ' ',
              pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

    discount_rate = .99
    epsilon = 0.1
    beta = .01
    tmax = 320
    SGD_epoch = 4

    # keep track of progress
    mean_rewards = []

    for e in range(episode):

        # collect trajectories
        old_probs, states, actions, rewards = \
            pong_utils.collect_trajectories(envs, policy, tmax=tmax)

        total_rewards = np.sum(rewards, axis=0)

        # gradient ascent step
        for _ in range(SGD_epoch):
            # uncomment to utilize your own clipped function!
            L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

            # L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        # the clipping parameter reduces as time goes on
        epsilon *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e + 1) % 20 == 0:
            print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
            print(total_rewards)

        # update progress widget bar
        timer.update(e + 1)

    timer.finish()