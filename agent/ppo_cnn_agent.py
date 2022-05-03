from torch.nn.modules.activation import ReLU
from agent.agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from utils.init_utils import init

learning_rate = 0.005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
warm_up = 100000

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class CNNA2C(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        super(CNNA2C, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(in_channels, 32, 8, strdie=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        self.fc = init_(nn.Linear(32 * 7 * 7, 512))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic = init_(nn.Linear(512, 1))
        self.actor = init_(nn.Linear(512, n_actions))

        self.train()

    def forward(self, x):
        x = F.relu(self.conv1(x / 255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc(x))
        
        value = self.critic(x)
        pi = self.actor(x)

        return value, pi

class PPOAgent(Agent):
    def __init__(self, observation_space:int, action_space:int, num_envs:int=1):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs

        self.experience_memory = []

        self.cnn_a2c = CNNA2C().to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        action_prob = self.actor(state)
        action = (Categorical(action_prob)).sample().item()
        return action, action_prob.detach()

    def save_xp(self, trajectory:tuple):
        self.experience_memory.append(trajectory)

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
            surr2 = torch.clamp(ratio_list, 1 - eps_clip, 1 + eps_clip).to(device) * advantage_list
            loss = -torch.min(surr1, surr2).to(device) + F.smooth_l1_loss(self.critic(state_list) , td_target.detach()).to(device)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        return loss.mean().detach().cpu().numpy()

    def save_model(self, save_dir:str):
        torch.save(self.actor.state_dict(), save_dir + "_actor.pt")
        torch.save(self.critic.state_dict(), save_dir + "_critic.pt")

    def load_model(self, load_dir:str):
        self.actor.load_state_dict(torch.load(load_dir + "_actor.pt"))
        self.critic.load_state_dict(torch.load(load_dir + "_critic.pt"))

