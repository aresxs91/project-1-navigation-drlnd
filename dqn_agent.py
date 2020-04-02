import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from memory import ReplayBuffer


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size (Initially 64)
GAMMA = 0.99           # discount factor (Initially 0.99)
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate (Initially 5e-4)
UPDATE_CYCLE = 4        # how many episodes need to pass for network updates to kick in

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, apply_dueling=False, apply_double=False):
        """
        Initialize a Unity agent object.
        :param state_size: (int) dimension of each state
        :param action_size: (int) dimension of each action
        :param seed: (int) random seed
        """
        assert(self._true_xor(apply_dueling, apply_double),
               "Choose one between dueling networks or DDQN")

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.apply_dueling = apply_dueling
        self.apply_double = apply_double

        # Q-Network
        self.q_net_target = QNetwork(state_size, action_size, seed, apply_dueling=apply_dueling).to(device)
        self.q_net_local = QNetwork(state_size, action_size, seed, apply_dueling=apply_dueling).to(device)
        self.opt = optim.Adam(self.q_net_local.parameters(), lr=LR)

        # Replay memory
        self.memory_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    @staticmethod
    def _true_xor(*args):
        return sum(args) == 1

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory buffer for future experience replay
        :param state: The current state of the agent
        :param action: The action that the agent has taken in given state
        :param reward: The reward associated with the state action combination
        :param next_state: The resulting state after taking action in previous state
        :param done: (bool) Has the terminal state been reached?
        :return: None
        """
        self.memory_buffer.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_CYCLE
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn from it
            if BATCH_SIZE < len(self.memory_buffer):
                experiences = self.memory_buffer.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like) current state
        :param eps: (float) epsilon, for epsilon-greedy action selection
        :return: (int) The index of the action to be taken by the agent
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_net_local.eval()
        with torch.no_grad():  # Do not perform a forward pass in this context
            action_values = self.q_net_local(state)
        self.q_net_local.train()

        # Epsilon-greedy action selection
        greed_p = random.random()

        return np.argmax(action_values.cpu().data.numpy()) if greed_p > eps else \
            random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples
        :param gamma: (float) discount factor
        :return:
        """
        states, actions, rewards, next_states, done_signals = experiences

        if not self.apply_double:
            # Get max predicted Q values for the next state of the target model.
            Q_targets_next = self.q_net_target(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            # In the case of Double-DQN, evaluate the best selected action with the target model's set of parameters.
            indices = torch.argmax(self.q_net_local(next_states).detach(), 1)  # The selected next best action's indices
            # Evaluate that action by comparing with the local network's set of parameters
            Q_targets_next = self.q_net_target(next_states).detach().gather(1, indices.unsqueeze(1))

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - done_signals))

        # Get expected Q values from local model (being trained)
        # x.gather(1, actions) returns a tensor which results from the concatenation of the input tensor values along
        # the given dimensions (here the dim indexes are the taken actions indices)
        Q_expected = self.q_net_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # perform network update
        self.soft_update(self.q_net_local, self.q_net_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Soft update model parameters, given by the function:
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :param tau: (float) interpolation parameter
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
