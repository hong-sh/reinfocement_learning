from agent.agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

learning_rate = 0.003
eps_clip = 0.1


device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class HiPPOAgent(Agent):
    def __init__(self, observation_space:int, high_level_action_space:int, low_level_action_space:int, num_envs:int=1):
        super().__init__()

        self.observation_space = observation_space
        self.high_level_action_space = high_level_action_space
        self.low_level_action_space = low_level_action_space
        self.num_envs = num_envs

        self.experience_memory = []
        if num_envs > 1:
            self.experience_memory = [[] for _ in range(num_envs)]

        self.high_level_actor = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Linear(32, high_level_action_space),
            nn.Softmax(dim=-1)
        )

        self.high_level_critic = nn.Sequential(
            nn.Linear(observation_space, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.low_level_actor = nn.Sequential(
            nn.Linear(observation_space+high_level_action_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, low_level_action_space),
            nn.Softmax(dim=-1)
        )

        self.low_level_critic = nn.Sequential(
            nn.Linear(observation_space+high_level_action_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.high_level_actor.to(device)
        self.high_level_critic.to(device)

        self.low_level_actor.to(device)
        self.low_level_critic.to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)



    def get_action(self, state:object):
        time_remaining = (self.period - self.count) / self.period
        if self.count % self.period == 0: # sample a new latent skill

        pass

    def save_xp(self, trajectory:tuple):
        pass

    def train(self):
        pass

    def save_model(self, save_dir:str):
        pass

    def load_model(self, load_dir:str):
        pass
