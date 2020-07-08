import numpy as np
import random
from collections import namedtuple, deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from model import QNetwork


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 1024 # 64         # minibatch size
GAMMA = 0.99            # discount factor
#TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 64        # how often to update the network
EPS_GREEDY = 0.1

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        tf.random.set_seed(seed)

        # Q-Network
        self.localActionModels, self.localFullModel = self.create_network(state_size, action_size)
        self.targetActionModels, self.targetFullModel = self.create_network(state_size, action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def create_network(self, state_size : int, action_size : int):
        inputs = keras.Input(shape=(state_size,), name="lunar_state")

        hiddenLayers = layers.Dense(6, activation="relu", name="hidden1")(inputs)
        hiddenLayers = layers.Dense(4, activation="relu", name="hidden2")(hiddenLayers)

        activationLayers = [layers.Dense(1, activation=keras.activations.linear, name="action_" + str(action))(hiddenLayers) for action in range(action_size)]

        # share the optimizer between different models so that they all share the same params
        optimizer = keras.optimizers.Adam()
        actionModels = []

        for action in range(action_size):

            actionModel = keras.models.Model(inputs=inputs, outputs=activationLayers[action], name="action_model_" + str(action))

            loss = 'mse'
            metrics = ['mae', 'mse']
            actionModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            actionModels.append(actionModel)

        fullModel = keras.layers.Concatenate()(activationLayers)

        return actionModels, fullModel

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=EPS_GREEDY):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        action_values = self.localFullModel([state])[0].numpy()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #states, actions, rewards, next_states, dones = experiences

        for state, action, reward, next_state, done in experiences:
            localActionModel : keras.Model = self.localActionModels[action]

            if done:
                y = reward
            else:
                y = reward + gamma * self.targetFullModel(next_state)[0].numpy().max()

            localActionModel.fit([state], [y], epochs=1)

        # ------------------- update target network ------------------- #
        self.update(self.localFullModel, self.targetFullModel)


    def update(self, local_model : keras.Sequential, target_model : keras.Sequential):
        #hard update instead
        target_model.set_weights(local_model.get_weights())

        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        #     target_param.data.copy_(tau*local_param.data + (1.0-TAU)*target_param.data)

    def save(self, pathPrefix : str):
        self.localFullModel.save_weights(pathPrefix + ".ckpt")
        for i in range(self.action_size):
            self.localActionModels[i].save_weights(pathPrefix + "_" + str(i) + ".ckpt")

    def load(self, pathPrefix : str):
        self.localFullModel.load_weights(pathPrefix + ".ckpt")
        for i in range(self.action_size):
            self.localActionModels[i].load_weights(pathPrefix + "_" + str(i) + ".ckpt")



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        return experiences

        '''
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        next_states = [e.next_state for e in experiences]
        dones = [e.done for e in experiences]
  
        return (states, actions, rewards, next_states, dones)
        '''

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)