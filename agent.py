import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1)

        # Calculates output size of convolutional operation
        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32 # total number of input nodes for fully-connected layer

        self.fc1 = nn.Linear(linear_input_size + 1, outputs)
        self.fc2 = nn.Linear(64, outputs)

    def forward(self, x, direction):
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        #x = F.relu(self.conv3(self.bn3(x)))
        #x = F.relu(self.fc1(torch.cat((x.view(x.size(0), -1), direction), dim=1)))
        #x = F.softmax(self.fc2(x), dim=1)
        x = F.softmax(self.fc1(torch.cat((x.view(x.size(0), -1), direction), dim=1)), dim=1)
        return x

class SnakeAgent:
    def __init__(self, w, h, n_actions):
        self.w = w
        self.h = h
        self.n_actions = n_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(w, h, n_actions)
        self.target_net = DQN(w, h, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.99
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TARGET_UPDATE = 10
        self.target_update_count = 0

    def action(self, state, t):
        board, direction = state

        weights = self.policy_net(board, direction)
        return torch.multinomial(weights, 1)

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-t / self.EPS_DECAY)

        if sample > eps_threshold: # greedy
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(board, direction).max(1)[1].view(1, 1)
        else: # random
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # Detailed explanation). This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Creates a list (mask) of true/false corresponding to whether or not the next state is terminal      
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        if len([s[0] for s in batch.next_state if s is not None]) == 0:
            return
        non_final_next_boards = torch.cat([s[0] for s in batch.next_state if s is not None])
        non_final_next_directions = torch.cat([s[1] for s in batch.next_state if s is not None])

        # Get data batches
        board_batch = torch.cat([s[0] for s in batch.state])
        direction_batch = torch.cat([s[1] for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(board_batch, direction_batch)

        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_boards, non_final_next_directions).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA).view(-1, 1) + reward_batch

        # Compute the loss
        loss = F.cross_entropy(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Copy policy net over to target net
        self.target_update_count = (self.target_update_count + 1) % self.TARGET_UPDATE
        if self.target_update_count == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_memory(self):
        return self.memory