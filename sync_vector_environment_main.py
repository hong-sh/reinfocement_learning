import gym
from gym.vector.sync_vector_env import SyncVectorEnv
import random
import numpy as np

from algorithm.ppo_agent import PPOAgent

import torch
from torch.distributions import Categorical

T_horizon = 100

if __name__ == "__main__":
    env = SyncVectorEnv([
        lambda: gym.make("CartPole-v1"),
        lambda: gym.make("CartPole-v1"),
        lambda: gym.make("CartPole-v1")
    ])
    agent = PPOAgent(observation_space=4, action_space=2, num_envs=env.num_envs)

    score = 0.0
    for i_eps in range(10000):
        states = env.reset()
        any_done = False
        while not any_done:
            
            for t in range(T_horizon):
                # for i in range(env.num_envs):
                actions, action_probs = agent.get_actions(states)

                next_states, rewards, dones, infos = env.step(actions)
                # print('next_state : {}, action : {}, reward : {}, done : {}, info : {}'.format(next_state, action, reward, done, info))

                for i in range(env.num_envs):
                    agent.save_xps(i, (states[i], next_states[i], actions[i], action_probs[i][actions[i]].item(), rewards[i], dones[i]))

                states = next_states
                score += np.mean(rewards)

                if np.any(dones):
                    any_done = True
                    break
            
            agent.train()
        if i_eps % 10 == 0:
            print("{} episode avg score : {:.1f}".format(i_eps+1, score/10))
            score = 0.0

        env.close()





