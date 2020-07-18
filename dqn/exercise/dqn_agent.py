import numpy as np
import random
from collections import namedtuple, deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.python.framework.ops import disable_eager_execution


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

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
        #disable_eager_execution()
        self.state_size = state_size
        self.action_size = action_size
        #tf.random.set_seed(seed)

        # Q-Network
        self.localModel = self.create_network(state_size, action_size)
        self.targetModel = self.create_network(state_size, action_size)

        self.optimizer = keras.optimizers.Adam(learning_rate=LR)
        self.loss_fn = keras.losses.MeanSquaredError()

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def create_network(self, state_size : int, action_size : int):
        inputs = keras.Input(shape=(state_size,), name="lunar_state")

        hiddenLayers = layers.Dense(64, activation="relu", name="hidden1")(inputs)
        hiddenLayers = layers.Dense(64, activation="relu", name="hidden2")(hiddenLayers)
        activationLayer = layers.Dense(action_size, activation="linear", name="activation")(hiddenLayers)

        return keras.models.Model(inputs=inputs, outputs=activationLayer, name="full_model")

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

    def act(self, state, eps=0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        action_values = self.localModel(state.reshape([1, self.state_size]))[0]

        # Epsilon-greedy action selectionÎ
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
        states, actions, rewards, next_states, dones = experiences

        Q_targets = rewards + (gamma * tf.reduce_max(self.targetModel(next_states), axis = 1) * (1 - dones))

        with tf.GradientTape() as tape:
            actionIndices = np.stack([np.array([i, action]) for i, action in enumerate(actions)])

            Q_expected_full = self.localModel(states, training=True)
            Q_expected = tf.gather_nd(Q_expected_full,actionIndices)

            loss_value = self.loss_fn(Q_targets, Q_expected)

        grads = tape.gradient(loss_value, self.localModel.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.localModel.trainable_variables))

        #print("done learning")
        # ------------------- update target network ------------------- #
        self.update(self.localModel, self.targetModel)


    def update(self, local_model : keras.Sequential, target_model : keras.Sequential):

        for t, l in zip(target_model.trainable_variables, local_model.trainable_variables):
            t.assign(t * (1 - TAU) + l * TAU)

    def save(self, pathPrefix : str):
        self.localModel.save_weights(pathPrefix + ".ckpt")

    def load(self, pathPrefix : str):
        self.localModel.load_weights(pathPrefix + ".ckpt")



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

        states = np.stack([e.state for e in experiences if e is not None])
        actions = np.stack([e.action for e in experiences if e is not None])
        rewards = np.stack([e.reward for e in experiences if e is not None])
        next_states = np.stack([e.next_state for e in experiences if e is not None])
        dones = np.stack([e.done for e in experiences if e is not None])
  
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)