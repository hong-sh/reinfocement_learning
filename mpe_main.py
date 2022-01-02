import supersuit
import pettingzoo.mpe.simple_tag_v2 as simple_tag_v2
import gym
import random
import numpy as np

import agent.ppo_agent as ppo

def get_action():
    return np.random.randint(0,5)

if __name__ == "__main__":
    env = simple_tag_v2.env(num_good=3, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
    env = gym.make('CartPole-v1')
    agent = ppo(env.observation_space.shape[0], env.action_space.shape[0])

    env.reset()
    step_cnt = 0
    for _ in range(1000):
        step_cnt += 1
        print('step cnt : {}'.format(step_cnt))
        action = agent.get_action()
        env.step(action)
        next_state, reward, done, info = env.last()
        env.render()
        print('next_state : {}, reward : {}, done : {}, info : {}'.format(next_state, reward, done, info))
