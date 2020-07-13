import numpy as np
import random
from collections import namedtuple, deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
        tf.random.set_seed(seed)

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

    def act(self, state, eps):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        action_values = self.localModel(state.reshape([1, self.state_size]))[0]

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
        #print("learning...")

        y_vals = [reward + gamma * self.targetModel(next_state.reshape([1, self.state_size]))[0].numpy().max() * done
                  for state, action, reward, next_state, done in experiences]
        '''
                experiencesPerAction = []
                y_valsPerAction = []
                for action in range(self.action_size):
                    experienceIndices = [i for i, experience in enumerate(experiences) if experience.action == action]
                    experiencesPerAction.append([experiences[i] for i in experienceIndices])
                    y_valsPerAction.append([y_vals[i] for i in experienceIndices])
        '''
        with tf.GradientTape() as tape:
            actionIndices = [[i, experience.action] for i, experience in enumerate(experiences)]

            fullPredictions = self.localModel(np.stack([state for state, _, _, _, _ in experiences]), training=True)
            predictions = tf.gather_nd(fullPredictions,actionIndices)

            loss_value = self.loss_fn(y_vals, predictions)

        grads = tape.gradient(loss_value, self.localModel.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.localModel.trainable_weights))

        #print("done learning")
        # ------------------- update target network ------------------- #
        self.update(self.localModel, self.targetModel)


    def update(self, local_model : keras.Sequential, target_model : keras.Sequential):
        #hard update instead
        #target_model.set_weights(local_model.get_weights())
        target_weights = target_model.get_weights()
        local_weights = local_model.get_weights()
        target_model.set_weights([target_weights[weightIndex] * (1 - TAU) + local_weights[weightIndex] * TAU for weightIndex in range(len(local_model.get_weights()))])

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
        self.localModel.save_weights(pathPrefix + ".ckpt")
        for i in range(self.action_size):
            self.localActionModels[i].save_weights(pathPrefix + "_" + str(i) + ".ckpt")

    def load(self, pathPrefix : str):
        self.localModel.load_weights(pathPrefix + ".ckpt")
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