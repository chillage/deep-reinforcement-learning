import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 6)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

network = QNetwork().to(device)

with torch.no_grad():

    network.eval()

    layers = [network.fc1, network.fc2]

    for layer in layers:
        weight = layer.weight
        for i in range(weight.size()[0]):
            for j in range(weight.size()[1]):
                weight[i, j] = ((i - 3) * 2) + pow((j - 1) * 1.2, 3)

        for i in range(layer.bias.size()[0]):
            layer.bias[i] = 2 - i

    input = torch.from_numpy(np.vstack([np.array([1, 2, 3]), np.array([-2,4,-6])])).float().unsqueeze(0).to(device)

    output = network(input).numpy()





    a = 5




