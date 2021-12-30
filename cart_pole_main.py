import gym
import random
import numpy as np

from algorithm.ppo_agent import PPOAgent

import torch
from torch.distributions import Categorical

T_horizon = 100

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = PPOAgent(observation_space=4, action_space=2)

    score = 0.0
    for i in range(10000):
        state = env.reset()
        done = False
        while not done:
            
            for t in range(T_horizon):
                action, action_prob = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                # print('next_state : {}, action : {}, reward : {}, done : {}, info : {}'.format(next_state, action, reward, done, info))

                agent.save_xp((state, next_state, action, action_prob[action].item(), reward, done))

                state = next_state
                score += reward
                if done:
                    break
            
            agent.train()
        if i % 10 == 0:
            print("{} episode avg score : {:.1f}".format(i+1, score/10))
            score = 0.0

        env.close()





