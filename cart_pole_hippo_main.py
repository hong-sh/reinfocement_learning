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

def random_hippo_run(env:object, log_name:str, num_iter:int, min_period:int, max_period:int):
    from agent.hippo_agent import HiPPOAgent
    random_hippo_sw = SummaryWriter('D:\\hong\\git_repo\\reinforcement_learning_algorithms\\logs\\' + log_name)
    hippo_random_agent = HiPPOAgent(observation_space=4, high_level_action_space=2, low_level_action_space=2, \
        min_period=min_period, max_period=max_period, random_period=True)

    step = 0
    for i in range(num_iter):
        hippo_random_agent.reset_count()
        state = env.reset()
        done = False
        score = 0.0
        l_score = 0.0
        episode_h_action_probs = []
        episode_l_action_probs = []
        episode_l_rewards = []

        while not done:
            for t in range(T_horizon):
                h_action, h_action_prob, l_action, l_action_prob = hippo_random_agent.get_action(state)
                next_state, h_reward, done, info = env.step(l_action)
                l_reward = calculate_low_level_reward(state, next_state, h_action, h_action_prob)

                mse = (np.square(max(h_action_prob) - max(l_action_prob))).mean()
                random_hippo_sw.add_scalar("ProbLoss", mse, step)
                episode_h_action_probs.append(h_action_prob)
                episode_l_action_probs.append(l_action_prob)
                episode_l_rewards.append(l_reward)

                hippo_random_agent.save_xp((state, next_state, h_action, h_action_prob, h_reward, l_action, l_action_prob, l_reward, done))

                state = next_state
                score += h_reward
                l_score += l_reward
                step += 1
                if done:
                    break
            hippo_random_agent.train()

        random_hippo_sw.add_scalar("Score", score, i+1)
        random_hippo_sw.add_scalar("LScore", l_score, i+1)
        random_hippo_sw.add_scalar("HProb", episode_h_action_probs, i+1)
        random_hippo_sw.add_scalar("LAction", episode_l_action_probs, i+1)
        random_hippo_sw.add_scalar("LReward", episode_l_rewards, i+1)
