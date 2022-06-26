from agent.agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils.distributions import Categorical
from utils.init_utils import init

learning_rate = 5e-4
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
entropy_coef = 0.001
value_clip = 0.2
value_loss_coef = 0.5
K_epoch = 4
max_grad_norm = 0.05

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class HiPPOCNNAgent(Agent):
    def __init__(self, num_inputs:int, high_level_action_space:int, low_level_action_space:int,\
         min_period:int, max_period:int, random_period:bool=False):
        super(HiPPOCNNAgent, self).__init__()

        self.num_inputs = num_inputs  
        self.high_level_action_space = high_level_action_space
        self.low_level_action_space = low_level_action_space

        self.min_period = min_period
        self.max_period = max_period
        self.random_period = random_period

        self.periods = np.arange(min_period, max_period + 1)
        self.curr_period = self.periods[0]
        self.max_period = max(self.periods)
        self.average_period = (min_period + max_period) / 2.0 if random_period else min_period

        self.experience_memory = []

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        self.fc = init_(nn.Linear(32*7*7, 512))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_high_v = init_(nn.Linear(512, 1))
        self.fc_high_pi = Categorical(512, high_level_action_space)

        self.fc_low_v = init_(nn.Linear(512+high_level_action_space, 1))
        self.fc_low_pi = Categorical(512+high_level_action_space, low_level_action_space)

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # self.train()

    def reset_count(self):
        self.count = 0

    def get_random_period(self):
        return self.periods[np.random.choice(len(self.periods))]

    def get_high_v(self, x):
        x = F.relu(self.conv1(x / 255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))

        value = self.fc_high_v(x)
        return value, x

    def get_low_v(self, x, high_level_action_logits):
        x = F.relu(self.conv1(x / 255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        x = torch.concat([x, high_level_action_logits], dim=1)
        value = self.fc_low_v(x)

        return value, x

    def get_high_pi(self, x):
        dist = self.fc_high_pi(x)
        action = dist.sample()
        action_log_probs = dist.log_probs(dist.sample())
        dist_entropy = dist.entropy().mean()
        return action, action_log_probs, dist_entropy, dist.logits

    def get_low_pi(self, x):
        dist = self.fc_low_pi(x)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return action, action_log_probs, dist_entropy

    def get_high_action(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(device)
                state = state.unsqueeze(0)

            value, x = self.get_high_v(state)
            action, action_log_probs, dist_entropy, dist_logits = self.get_high_pi(x)

        return action.view(-1).detach().numpy()[0], action_log_probs.view(-1).detach().numpy()[0], dist_logits.detach()

    def get_low_action(self, state, high_level_action_logits):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(device)
                state = state.unsqueeze(0)

            value, x = self.get_low_v(state, high_level_action_logits)
            action, action_log_probs, dist_entropy = self.get_low_pi(x)

        return action.view(-1).detach().numpy()[0], action_log_probs.view(-1).detach().numpy()[0]

    def get_action(self, state:object):
        if self.count % self.curr_period == 0:
            if self.random_period:
                self.curr_period = self.get_random_period()
            time_remaining = (self.curr_period - self.count) / self.curr_period
            high_level_action, high_level_action_prob, high_level_action_logits = self.get_high_action(state)
            self.curr_high_level_action = high_level_action
            self.curr_high_level_action_prob = high_level_action_prob
            self.curr_high_level_action_logits = high_level_action_logits

        self.count = (self.count + 1) % self.curr_period
        low_level_action, low_level_action_prob = self.get_low_action(state, self.curr_high_level_action_logits)
        return self.curr_high_level_action, self.curr_high_level_action_prob, self.curr_high_level_action_logits.view(-1).detach().numpy(), \
            low_level_action, low_level_action_prob

    def evaluate_high_actions(self, states, actions):
        value, x = self.get_high_v(states)
        dist = self.fc_high_pi(x)
        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

    def evaluate_low_actions(self, states, high_action_logits, low_actions):
        value, x = self.get_low_v(states, high_action_logits)
        dist = self.fc_low_pi(x)
        action_log_probs = dist.log_probs(low_actions)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

    def save_xp(self, trajectory:tuple):
        self.experience_memory.append(trajectory)

    def make_batch(self):
        state_list, next_state_list, h_action_list, h_action_prob_list, h_action_logits_list, h_reward_list, \
            l_action_list, l_action_prob_list, l_reward_list, done_list =\
                 [], [], [], [], [], [], [], [], [], []

        for experience in self.experience_memory:
            state, next_state, h_action, h_action_prob, h_action_logits, h_reward,\
                l_action, l_action_prob, l_reward, done = experience

            state_list.append(state)
            next_state_list.append(next_state)
            h_action_list.append([h_action])
            h_action_prob_list.append([h_action_prob])
            h_action_logits_list.append(h_action_logits)
            h_reward_list.append([h_reward])
            l_action_list.append([l_action])
            l_action_prob_list.append([l_action_prob])
            l_reward_list.append([l_reward])
            done = 0 if done else 1
            done_list.append([done])

        state_list, next_state_list, h_action_list, h_action_prob_list, h_action_logits_list, h_reward_list, \
            l_action_list, l_action_prob_list, l_reward_list, done_list = \
            torch.tensor(state_list, dtype=torch.float).to(device), torch.tensor(next_state_list, dtype=torch.float).to(device), \
                torch.tensor(h_action_list).to(device), torch.tensor(h_action_prob_list).to(device), torch.tensor(h_action_logits_list).to(device), torch.tensor(h_reward_list).to(device), \
                        torch.tensor(l_action_list).to(device), torch.tensor(l_action_prob_list).to(device), torch.tensor(l_reward_list).to(device), \
                            torch.tensor(done_list, dtype=torch.float).to(device)
        
        self.experience_memory = []
        return state_list, next_state_list, h_action_list, h_action_prob_list, h_action_logits_list, h_reward_list, \
            l_action_list, l_action_prob_list, l_reward_list, done_list

    def train(self):
        state_list, next_state_list, h_action_list, h_action_prob_list, h_action_logits_list, h_reward_list, \
            l_action_list, l_action_prob_list, l_reward_list, done_list = self.make_batch()

        h_pi_loss_mean, h_value_loss_mean, h_dist_entropy_mean = 0.0, 0.0, 0.0
        l_pi_loss_mean, l_value_loss_mean, l_dist_entropy_mean = 0.0, 0.0, 0.0
        h_approx_kl_mean, h_approx_ent_mean, h_clipfrac_mean = 0.0, 0.0, 0.0
        l_approx_kl_mean, l_approx_ent_mean, l_clipfrac_mean = 0.0, 0.0, 0.0

        for _ in range(K_epoch):
            # high level policy update
            h_value_next, _ = self.get_high_v(next_state_list)
            h_value, h_action_log_probs, h_dist_entropy = self.evaluate_high_actions(state_list, h_action_list)

            h_td_target = h_reward_list + gamma * h_value_next * done_list
            h_delta = h_td_target - h_value
            h_delta = h_delta.detach().cpu().numpy()

            h_advantage_list = []
            h_advantage = 0.0
            for delta_t in h_delta[::-1]:
                h_advantage = gamma * lmbda * h_advantage + delta_t[0]
                h_advantage_list.append([h_advantage])
            h_advantage_list.reverse()
            h_advantage_list = torch.tensor(h_advantage_list, dtype=torch.float).to(device)

            h_ratio_list = torch.exp(h_action_log_probs - h_action_prob_list).to(device)

            h_surr1 = h_ratio_list * h_advantage_list
            h_surr2 = torch.clamp(h_ratio_list, 1 - eps_clip, 1 + eps_clip).to(device) * h_advantage_list
            h_pi_loss = -torch.min(h_surr1, h_surr2).mean()

            h_value_pred_clipped = h_value_next + (h_value - h_value_next).clamp(-value_clip, value_clip)
            h_td_target = h_td_target.detach()
            h_value_losses = (h_value - h_td_target).pow(2)
            h_value_losses_clipped = (h_value_pred_clipped - h_td_target).pow(2)
            h_value_loss = 0.5 * torch.max(h_value_losses, h_value_losses_clipped).mean()

            # low level policy update
            l_value_next, _ = self.get_low_v(next_state_list, h_action_logits_list)
            l_value, l_action_log_probs, l_dist_entropy = self.evaluate_low_actions(state_list, h_action_logits_list, l_action_list)

            l_td_target = l_reward_list + gamma * l_value_next * done_list
            l_delta = l_td_target - l_value
            l_delta = l_delta.detach().cpu().numpy()

            l_advantage_list = []
            l_advantage = 0.0
            for delta_t in l_delta[::-1]:
                l_advantage = gamma * lmbda * l_advantage + delta_t[0]
                l_advantage_list.append([l_advantage])
            l_advantage_list.reverse()
            l_advantage_list = torch.tensor(l_advantage_list, dtype=torch.float).to(device)

            l_ratio_list = torch.exp(l_action_log_probs - l_action_prob_list).to(device)

            l_surr1 = l_ratio_list * l_advantage_list
            l_surr2 = torch.clamp(l_ratio_list, 1 - eps_clip, 1 + eps_clip).to(device) * l_advantage_list
            l_pi_loss = -torch.min(l_surr1, l_surr2).mean()

            l_value_pred_clipped = l_value_next + (l_value - l_value_next).clamp(-value_clip, value_clip)
            l_td_target = l_td_target.detach()
            l_value_losses = (l_value - l_td_target).pow(2)
            l_value_losses_clipped = (l_value_pred_clipped - l_td_target).pow(2)
            l_value_loss = 0.5 * torch.max(l_value_losses, l_value_losses_clipped).mean()

            h_loss = h_value_loss * value_loss_coef + h_pi_loss - h_dist_entropy * entropy_coef
            l_loss = l_value_loss * value_loss_coef + l_pi_loss - l_dist_entropy * entropy_coef
            total_loss = h_loss / self.average_period + l_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()

            h_pi_loss_mean += h_pi_loss.item()
            h_value_loss_mean += h_value_loss.item()
            h_dist_entropy_mean += h_dist_entropy.item()

            l_pi_loss_mean += l_pi_loss.item()
            l_value_loss_mean += l_value_loss.item()
            l_dist_entropy_mean += l_dist_entropy.item()

            h_approx_kl = h_ratio_list.mean()
            h_approx_ent = (-h_action_prob_list).mean()
            h_clipped = torch.where(torch.abs(h_ratio_list - 1.0) > eps_clip)
            h_clipfrac = len(h_clipped) / len(h_ratio_list)

            l_approx_kl = l_ratio_list.mean()
            l_approx_ent = (-l_action_prob_list).mean()
            l_clipped = torch.where(torch.abs(l_ratio_list - 1.0) > eps_clip)
            l_clipfrac = len(l_clipped) / len(l_ratio_list)

            h_approx_kl_mean += h_approx_kl
            h_approx_ent_mean += h_approx_ent
            h_clipfrac_mean += h_clipfrac

            l_approx_kl_mean += l_approx_kl
            l_approx_ent_mean += l_approx_ent
            l_clipfrac_mean += l_clipfrac


        h_pi_loss_mean /= K_epoch
        h_value_loss_mean /= K_epoch
        h_dist_entropy_mean /= K_epoch

        h_approx_kl_mean /= K_epoch
        h_approx_ent_mean /= K_epoch
        h_clipfrac_mean /= K_epoch

        l_pi_loss_mean /= K_epoch
        l_value_loss_mean /= K_epoch
        l_dist_entropy_mean /= K_epoch

        l_approx_kl_mean /= K_epoch
        l_approx_ent_mean /= K_epoch
        l_clipfrac_mean /= K_epoch

        return h_pi_loss_mean, h_value_loss_mean, h_dist_entropy_mean, h_approx_kl_mean, h_approx_ent_mean, h_clipfrac_mean, \
            l_pi_loss_mean, l_value_loss_mean, l_dist_entropy_mean, l_approx_kl_mean, l_approx_ent_mean, l_clipfrac_mean

    def save_model(self, save_dir:str):
        pass
        # torch.save(self.actor.state_dict(), save_dir + "_actor.pt")
        # torch.save(self.critic.state_dict(), save_dir + "_critic.pt")

    def load_model(self, load_dir:str):
        pass
        # self.actor.load_state_dict(torch.load(load_dir + "_actor.pt"))
        # self.critic.load_state_dict(torch.load(load_dir + "_critic.pt"))

