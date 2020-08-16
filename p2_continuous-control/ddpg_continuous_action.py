# based on : https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py

# Reference: https://arxiv.org/pdf/1509.02971.pdf
# https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from unityagents import UnityEnvironment

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import time
import random
import os

from helpers import ReplayBuffer, QNetwork, Actor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                        help='the maximum length of each episode')
    parser.add_argument('--total-episodes', type=int, default=2000,
                        help='total episodes of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--wandb-api-key', type=str, default=None,
                        help="the wandb API key")


    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=int(1e6),
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=256,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--exploration-noise', type=float, default=0.1,
                        help='the scale of exploration noise')
    parser.add_argument('--learning-starts', type=int, default=25e3,
                        help="timestep to start learning")
    parser.add_argument('--policy-frequency', type=int, default=5,
                        help="the frequency of training policy (delayed)")
    parser.add_argument('--noise-clip', type=float, default=0.5,
                        help='noise clip parameter of the Target Policy Smoothing Regularization')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    if args.wandb_api_key is not None:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key

    import wandb

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
               name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

#env = gym.make(args.gym_id)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic




uenv = UnityEnvironment(file_name='Reacher_Linux_Multi/Reacher.x86_64')
brain_name = uenv.brain_names[0]
brain = uenv.brains[brain_name]
# reset the environment
env_info = uenv.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size # number of actions
state_size = env_info.vector_observations.shape[1] # number of observations/states

num_agents = len(env_info.agents)
action_min = -1.0
action_max = 1.0


rb = ReplayBuffer(args.buffer_size)
actor = Actor(device, action_size, state_size).to(device)
qf1 = QNetwork(device, action_size, state_size).to(device)
qf1_target = QNetwork(device, action_size, state_size).to(device)
target_actor = Actor(device, action_size, state_size).to(device)
target_actor.load_state_dict(actor.state_dict())
qf1_target.load_state_dict(qf1.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()
# TRY NOT TO MODIFY: start the game
env_info = uenv.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
scores = np.zeros(num_agents)
episode_num = 0
global_step = 0

while True:
    global_step += 1
    # ALGO LOGIC: put action logic here
    if global_step < args.learning_starts:
        actions = [action for action in np.random.randn(num_agents, action_size)]
    else:
        actions = [actor.forward(obs.reshape((1,) + state_size)) for obs in env_info.vector_observations]
        actions = [(action.tolist()[0]
                + np.random.normal(0, action_max * args.exploration_noise, size=action_size)
        ).clip(action_min, action_max) for action in actions]

    env_info = uenv.step(actions)[brain_name]
    next_states = env_info.vector_observations  # get next state (for each agent)
    rewards = env_info.rewards  # get reward (for each agent)
    rewards = [0.1 if r > 0 else 0 for r in rewards]  # tweak for compatibility with this version of the env
    dones = env_info.local_done

    scores += rewards

    # ALGO LOGIC: training.
    rb.put(zip(states, actions, rewards, next_states, dones))
    if global_step > args.learning_starts:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_actions = (
                target_actor.forward(s_next_obses)
            ).clamp(action_min, action_max)
            qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions)
            next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args.gamma * (
                qf1_next_target).view(-1)

        qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device)).view(-1)
        qf1_loss = loss_fn(qf1_a_values, next_q_value)

        # optimize the midel
        q_optimizer.zero_grad()
        qf1_loss.backward()
        nn.utils.clip_grad_norm_(list(qf1.parameters()), args.max_grad_norm)
        q_optimizer.step()

        if global_step % args.policy_frequency == 0:
            actor_loss = -qf1.forward(s_obs, actor.forward(s_obs)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
            actor_optimizer.step()

            # update the target network
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        #if global_step % 100 == 0:
        #    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
        #    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    states = next_states

    if np.any(dones):
        episode_num += 1
        average_score = np.mean(scores)
        print(f"Total score (averaged over agents) for episode {episode_num} :\t {average_score}")
        writer.add_scalar("charts/episode_reward", average_score, episode_num)
        obs, scores = uenv.reset(train_mode=True)[brain_name], np.zeros(num_agents)

        if episode_num >= args.total_episodes:
            break

uenv.close()
writer.close()
