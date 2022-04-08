from agent.agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.distributions import OneHotCategorical

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
    def __init__(self, observation_space:int, high_level_action_space:int, low_level_action_space:int):
        super().__init__()

        self.observation_space = observation_space
        self.high_level_action_space = high_level_action_space
        self.low_level_action_space = low_level_action_space

        self.experience_memory = []

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

    def get_high_level_action(self, state:object):
        action_prob = self.high_level_actor(state)
        action = OneHotCategorical(action_prob).sample().item()
        return action, action_prob

    def get_low_level_action(self, state:object, high_level_action:int):
        state = torch.concat([state, high_level_action])
        action_prob = self.low_level_actor(state)
        action = (Categorical(action_prob)).sample().item()
        return action, action_prob


    def get_action(self, state:object):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)

        time_remaining = (self.period - self.count) / self.period
        if self.count % self.period == 0: # sample a new latent skill
            high_level_action, high_level_action_prob = self.get_high_level_action(state)
            self.curr_high_level_action = high_level_action
            self.curr_high_level_action_prob = high_level_action_prob

        self.count = (self.count + 1) % self.period
        low_level_action, low_level_action_prob = self.get_low_level_action(state, self.curr_high_level_action)
        return self.curr_high_level_action, self.curr_high_level_action_prob, low_level_action, low_level_action_prob        

    def save_xp(self, trajectory:tuple):
        self.experience_memory.append(trajectory)

    def make_batch(self):
        state_list, next_state_list, h_action_list, h_action_prob_list, h_reward_list, \
            l_action_list, l_action_prob_list, l_reward_list, done_list = [], [], [], [], [], [], [], [], []

        for experience in self.experience_memory:
            state, next_state, h_action, h_action_prob, h_reward, l_action, l_action_prob, l_reward, done = experience

            state_list.append(state)
            next_state_list.append(next_state)
            h_action_list.append(h_action)
            h_action_prob_list.append(h_action_prob)
            h_reward_list.append([h_reward])
            l_action_list.append(l_action)
            l_action_prob_list.append(l_action_prob)
            l_reward_list.append([l_reward])
            done_list.append([done])

        state_list, next_state_list, h_action_list, h_action_prob_list, h_reward_list, \
            l_action_list, l_action_prob_list, l_reward_list, done_list = \
            torch.tensor([state_list], dtype=torch.float).to(device), torch.tensor([next_state_list], dtype=torch.float).to(device), \
                torch.tensor([h_action_list]).to(device), torch.tensor([h_action_prob_list]).to(device), torch.tensor([h_reward_list]).to(device), \
                    torch.tensor([h_action_list]).to(device), torch.tensor([h_action_prob_list]).to(device), torch.tensor([h_reward_list]).to(device), \
                        torch.tensor([done_list], dtype=torch.float).to(device)
        
        self.experience_memory = []
        return state_list, next_state_list, h_action_list, h_action_prob_list, h_reward_list, l_action_list, l_action_prob_list, l_reward_list, done_list

    def train(self):
        pass

    def save_model(self, save_dir:str):
        pass

    def load_model(self, load_dir:str):
        pass
