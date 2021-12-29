from torch.nn.modules.activation import ReLU
from agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

learning_rate = 0.001

class PPOAgent(Agent):
    def __init__(self, observation_space:int, action_space:int):
        super(PPOAgent).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.experience_memory = []

        self.actor = nn.Sequential(
            nn.Linear(observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(observation_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


    def get_action(self):
        pass

    def save_xp(self, trajectory):
        pass

    def training(self):
        pass