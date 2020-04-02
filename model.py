import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, apply_dueling, fc1_units=128, fc2_units=128):
        """
        Initialize parameters and build model.
        :param state_size: (int) Dimension of each state
        :param action_size: (int) Dimension of each action
        :param seed: (int) Random seed
        :param apply_dueling: (bool) Whether to use dueling networks or not
        :param fc1_units: Number of nodes in first hidden layer
        :param fc2_units: Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.apply_dueling = apply_dueling

        fc1_units = fc1_units
        fc2_units = fc2_units
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.state_value = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """
        Perform a forward pass through the neural network to map states to action values
        :param state: The current state of the agent
        :return: A vector of probabilities for all action values.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        if self.apply_dueling:
            # advantage values + state value
            return self.fc3(x) + self.state_value(x)
        else:
            x = self.fc3(x)
        
        return x
