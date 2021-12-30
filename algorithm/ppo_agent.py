from torch.nn.modules.activation import ReLU
from algorithm.agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

learning_rate = 0.005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class PPOAgent(Agent):
    def __init__(self, observation_space:int, action_space:int):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.experience_memory = []

        self.actor = nn.Sequential(
            nn.Linear(observation_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(observation_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.actor.to(device)
        self.critic.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        action_prob = self.actor(state)
        action = (Categorical(action_prob)).sample().item()
        return action, action_prob.detach()

    def save_xp(self, trajectory):
        self.experience_memory.append(trajectory)

    def train(self):
        state_list, next_state_list, action_list, action_prob_list, reward_list, done_list = [], [], [], [], [], []
        for experience in self.experience_memory:
            state, next_state, action, action_prob, reward, done = experience

            state_list.append(state)
            next_state_list.append(next_state)
            action_list.append([action])
            action_prob_list.append([action_prob])
            reward_list.append([reward])
            done = 0 if done else 1
            done_list.append([done])

        # print('state_list : ', state_list)
        # print('next_state_list : ', next_state_list)
        # print('action_list : ', action_list)
        # print('action_prob_list : ', action_prob_list)
        # print('reward_list : ', reward_list)
        # print('done_list : ', done_list)

        state_list, next_state_list, action_list, action_prob_list, reward_list, done_list = \
            torch.tensor(state_list, dtype=torch.float).to(device), torch.tensor(next_state_list, dtype=torch.float).to(device), \
                torch.tensor(action_list).to(device), torch.tensor(action_prob_list).to(device), torch.tensor(reward_list).to(device), torch.tensor(done_list, dtype=torch.float).to(device)
        
        # Normalized Reward
        # reward_list = (reward_list - reward_list.mean()) / (reward_list.std() + 1e-2)

        self.experience_memory = []

        td_target = reward_list + gamma * self.critic(next_state_list) * done_list
        delta = td_target - self.critic(state_list)
        delta = delta.detach().cpu().numpy()

        advantage_list = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_list.append([advantage])
        advantage_list.reverse()
        advantage_list = torch.tensor(advantage_list, dtype=torch.float).to(device)

        pi_list = self.actor(state_list)
        pi_a_list = pi_list.gather(1, action_list)
        ratio_list = torch.exp(torch.log(pi_a_list) - torch.log(action_prob_list)).to(device)

        surr1 = ratio_list * advantage_list
        surr2 = torch.clamp(ratio_list, 1 - eps_clip, 1+eps_clip).to(device) * advantage_list
        loss = -torch.min(surr1, surr2).to(device) + F.smooth_l1_loss(self.critic(state_list) , td_target.detach()).to(device)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
