import gym
import random
import numpy as np
import math

import torch
from torch.distributions import Categorical

from tensorboardX import SummaryWriter

T_horizon = 100

def calculate_low_level_reward(state, next_state, h_action, h_action_prob):
    prev_pole_angle, curr_pole_angle = state[2], next_state[2]
    if h_action == 0:
        l_reward = 0.1 if prev_pole_angle < curr_pole_angle else -0.1
    else:
        l_reward = 0.1 if prev_pole_angle > curr_pole_angle else -0.1

    return l_reward

    