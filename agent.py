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

class ValueNet(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        #self.bn1 = nn.BatchNorm2d(16)
        #self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        #self.bn2 = nn.BatchNorm2d(32)

        # Calculates output size of convolutional operation
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=0):
            return (size - (kernel_size - 1) - 1 + 2*padding) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32 # total number of input nodes for fully-connected layer

        self.fc1 = nn.Linear(linear_input_size, 64)
        self.fc2 = nn.Linear(64, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x

class SnakeCritic:
    def __init__(self, w, h, n_actions):
        self.w = w
        self.h = h
        self.n_actions = n_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.value_net = ValueNet(w, h, n_actions)
        self.target_net = ValueNet(w, h, n_actions)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.value_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory(10000)
        
        self.BATCH_SIZE = 64
        self.GAMMA = 0.95
        # self.EPS_START = 0.99
        # self.EPS_END = 0.00
        # self.EPS_DECAY = 3000
        self.TARGET_UPDATE = 1000
        self.target_update_count = 0

    # def action(self, state, t):
    #     sample = random.random()
    #     eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-t / self.EPS_DECAY)

    #     if sample > eps_threshold: # greedy
    #         with torch.no_grad():
    #             return self.value_net(state).max(1)[1].view(1, 1)
    #     else: # random
    #         return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Creates a list (mask) of true/false corresponding to whether or not the next state is terminal      
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        if len([s[0] for s in batch.next_state if s is not None]) == 0:
            return
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Get data batches
        states_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.value_net(states_batch).gather(1, action_batch)

        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA).view(-1, 1) + reward_batch

        # Compute the loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Copy value net over to target net
        self.target_update_count = (self.target_update_count + 1) % self.TARGET_UPDATE
        if self.target_update_count == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

class PolicyNet(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

        # Calculates output size of convolutional operation
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=0):
            return (size - (kernel_size - 1) - 1 + 2*padding) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32 # total number of input nodes for fully-connected layer

        self.fc1 = nn.Linear(linear_input_size, 64)
        self.fc2 = nn.Linear(64, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class SnakeActor:
    def __init__(self, w, h, n_actions):
        self.w = w
        self.h = h
        self.n_actions = n_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PolicyNet(w, h, n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0001)
        self.GAMMA = 0.95

    def action(self, state):
        return torch.multinomial(self.policy(state), 1)

    def optimize(self, state_action_value, expected_state_action_value):
        #state_action_values = self.policy(state)
        #expected_state_action_values = reward + self.GAMMA * self.policy(next_state)
        
        # Compute the loss
        loss = F.cross_entropy(state_action_value, expected_state_action_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

class SnakeAgent:
    def __init__(self, w, h, n_actions):
        self.actor = SnakeActor(w, h, n_actions)
        self.critic = SnakeCritic(w, h, n_actions)

        self.GAMMA = 0.95

    def action(self, state):
        return self.actor.action(state)

    def optimize(self, state, action, next_state, reward):
        self.critic.memory.push(state, action, next_state, reward)
        self.critic.optimize()

        # Critic evaluates Q(s,a) for each a, given input s
        state_action_value = self.critic.value_net(state)
        expected_state_action_value = reward + self.GAMMA * self.critic.value_net(next_state) if next_state != None else 0

        self.actor.optimize(state_action_value, expected_state_action_value.argmax().view(1))



    