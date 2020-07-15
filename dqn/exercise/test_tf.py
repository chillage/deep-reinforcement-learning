import numpy as np
import random
from collections import namedtuple, deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(3,), name="lunar_state")

hiddenLayers = layers.Dense(5, activation="relu", name="hidden1")(inputs)
hiddenLayers = layers.Dense(4, activation="relu", name="hidden2")(hiddenLayers)
activationLayer = layers.Dense(6, activation="linear", name="activation")(hiddenLayers)

model = keras.models.Model(inputs=inputs, outputs=activationLayer, name="full_model")

for layer in model.layers[1:]:
    layerWeights = layer.get_weights()
    weight = layerWeights[0]
    bias = layerWeights[1]

    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            weight[i, j] = ((i - 3) * 2) + pow((j - 1) * 1.2, 3)

    for i in range(bias.shape[0]):
        bias[i] = 2 - i

    layer.set_weights(layerWeights)

input = np.vstack([np.array([1, 2, 3]), np.array([-2,4,-6])])

output = model(input).numpy()

a = 5