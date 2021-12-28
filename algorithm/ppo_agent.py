from agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 0.001

class PPOAgent(Agent):
    def __init__(self, observation_space:int, action_space:int):
        super(PPOAgent).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.observation_space, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


    def get_action(self):
        pass

    def save_xp(self, trajectory):
        pass

    def training(self):
        pass