from agent.agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils.distributions import Categorical
from utils.init_utils import init

learning_rate = 7e-4
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
entropy_coef = 0.001
value_clip = 0.2
value_loss_coef = 0.5
K_epoch = 3
max_grad_norm = 0.05

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class PPOCNNAgent(Agent):
    def __init__(self, num_inputs:int, action_space:int):
        super(PPOCNNAgent, self).__init__()

        self.num_inputs = num_inputs  
        self.action_space = action_space

        self.experience_memory = []

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        self.fc = init_(nn.Linear(32*7*7, 512))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_v = init_(nn.Linear(512, 1))
        self.fc_pi = Categorical(512, action_space)

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # self.train()

    def get_v(self, x):
        x = F.relu(self.conv1(x / 255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))

        value = self.fc_v(x)
        return value, x

    def get_pi(self, x):
        dist = self.fc_pi(x)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return action, action_log_probs, dist_entropy

    def get_action(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(device)
                state = state.unsqueeze(0)

            value, x = self.get_v(state)
            action, action_log_probs, dist_entropy = self.get_pi(x)

        return action.view(-1).detach().numpy()[0], action_log_probs.view(-1).detach().numpy()[0]

    def evaluate_actions(self, states, actions):
        value, x = self.get_v(states)
        dist = self.fc_pi(x)
        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

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
            torch.tensor(state_list, dtype=torch.float).to(device), torch.tensor(next_state_list, dtype=torch.float).to(device), \
                torch.tensor(action_list).to(device), torch.tensor(action_prob_list).to(device), torch.tensor(reward_list).to(device), torch.tensor(done_list, dtype=torch.float).to(device)
        
        self.experience_memory = []
        return state_list, next_state_list, action_list, action_prob_list, reward_list, done_list

    def train(self):
        state_list, next_state_list, action_list, action_prob_list, reward_list, done_list = self.make_batch()

        pi_loss_mean, value_loss_mean, dist_entropy_mean = 0.0, 0.0, 0.0

        for _ in range(K_epoch):
            value_next, _ = self.get_v(next_state_list)
            value, action_log_probs, dist_entropy = self.evaluate_actions(state_list, action_list)

            td_target = reward_list + gamma * value_next * done_list
            delta = td_target - value
            delta = delta.detach().cpu().numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage_list = torch.tensor(advantage_list, dtype=torch.float).to(device)

            ratio_list = torch.exp(action_log_probs - action_prob_list).to(device)

            surr1 = ratio_list * advantage_list
            surr2 = torch.clamp(ratio_list, 1 - eps_clip, 1 + eps_clip).to(device) * advantage_list
            pi_loss = -torch.min(surr1, surr2).mean()

            value_pred_clipped = value_next + (value - value_next).clamp(-value_clip, value_clip)
            td_target = td_target.detach()
            value_losses = (value - td_target).pow(2)
            value_losses_clipped = (value_pred_clipped - td_target).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            approx_kl = ratio_list.mean()
            approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
            approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
            clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
            clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

            self.optimizer.zero_grad()
            (value_loss * value_loss_coef + pi_loss - dist_entropy * entropy_coef).backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()

            pi_loss_mean = pi_loss.item()
            value_loss_mean = value_loss.item()
            dist_entropy_mean = dist_entropy.item()

        pi_loss_mean /= K_epoch
        value_loss_mean /= K_epoch
        dist_entropy_mean /= K_epoch

        return pi_loss_mean, value_loss_mean, dist_entropy_mean

    def save_model(self, save_dir:str):
        pass
        # torch.save(self.actor.state_dict(), save_dir + "_actor.pt")
        # torch.save(self.critic.state_dict(), save_dir + "_critic.pt")

    def load_model(self, load_dir:str):
        pass
        # self.actor.load_state_dict(torch.load(load_dir + "_actor.pt"))
        # self.critic.load_state_dict(torch.load(load_dir + "_critic.pt"))

