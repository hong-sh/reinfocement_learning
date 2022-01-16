from torch.nn.modules.activation import ReLU
from agent.agent import Agent

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

class PPOLSTMAgent(Agent):
    def __init__(self, observation_space:int, action_space:int, num_envs:int=1):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs

        self.experience_memory = []
        if num_envs > 1:
            self.experience_memory = [[] for _ in range(num_envs)]
        
        self.hidden_buffer = (torch.zeros([1, 1, 32], dtype=torch.float), \
            torch.zeros([1, 1, 32], dtype=torch.float))
        self.fc1 = nn.Linear(observation_space, 64)
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32, action_space)
        self.fc_v = nn.Linear(32, 1)
        
        self.fc1.to(device)
        self.lstm.to(device)
        self.fc_pi.to(device)
        self.fc_v.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        x = F.relu(self.fc1(state))
        print(x.size())
        x = x.view(-1, 1, 64)
        print(x.size())
        x, hidden = self.lstm(x, self.hidden_buffer)
        x = self.fc_pi(x)
        action_prob = F.softmax(x, dim=-1)
        print(x.size())
        action = (Categorical(action_prob)).sample().item()
        return action, action_prob.detach(), hidden
    

    def get_actions(self, states):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float().to(device)
        action_probs = self.actor(states)
        actions = (Categorical(action_probs)).sample()
        actions = actions.detach().cpu().numpy()
        return actions, action_probs.detach()

    def save_xp(self, trajectory:tuple):
        self.experience_memory.append(trajectory)

    def save_xps(self, index:int, trajectory:tuple):
        self.experience_memory[index].append(trajectory)

    def make_batch(self):
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

        state_list, next_state_list, action_list, action_prob_list, reward_list, done_list = \
            torch.tensor([state_list], dtype=torch.float).to(device), torch.tensor([next_state_list], dtype=torch.float).to(device), \
                torch.tensor([action_list]).to(device), torch.tensor([action_prob_list]).to(device), torch.tensor([reward_list]).to(device), torch.tensor([done_list], dtype=torch.float).to(device)
        
        self.experience_memory = []
        return state_list, next_state_list, action_list, action_prob_list, reward_list, done_list

    def make_batchs(self):
        state_batch, next_state_batch, action_batch, action_prob_batch, reward_batch, done_batch = [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)],\
            [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            state_list, next_state_list, action_list, action_prob_list, reward_list, done_list = [], [], [], [], [], []
            for experience in self.experience_memory[i]:
                state, next_state, action, action_prob, reward, done = experience

                state_list.append(state)
                next_state_list.append(next_state)
                action_list.append([action])
                action_prob_list.append([action_prob])
                reward_list.append([reward])
                done = 0 if done else 1
                done_list.append([done])

            state_batch[i] = state_list
            next_state_batch[i] = next_state_list
            action_batch[i] = action_list
            action_prob_batch[i] = action_prob_list
            reward_batch[i] = reward_list
            done_batch[i] = done_list

        state_batch, next_state_batch, action_batch, action_prob_batch, reward_batch, done_batch = \
            torch.tensor(state_batch, dtype=torch.float).to(device), torch.tensor(next_state_batch, dtype=torch.float).to(device), \
                torch.tensor(action_batch).to(device), torch.tensor(action_prob_batch).to(device), torch.tensor(reward_batch).to(device), torch.tensor(done_batch, dtype=torch.float).to(device)
        
        # Normalized Reward
        # reward_list = (reward_list - reward_list.mean()) / (reward_list.std() + 1e-2)

        if self.num_envs > 1:
            self.experience_memory = [[] for _ in range(self.num_envs)]
        return state_batch, next_state_batch, action_batch, action_prob_batch, reward_batch, done_batch

    def train(self):
        state_batch, next_state_batch, action_batch, action_prob_batch, reward_batch, done_batch = self.make_batch() if self.num_envs == 1 else self.make_batchs()
        for i in range(self.num_envs):
            state_list, next_state_list, action_list, action_prob_list, reward_list, done_list = state_batch[i], next_state_batch[i], action_batch[i], action_prob_batch[i], reward_batch[i], done_batch[i]
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
