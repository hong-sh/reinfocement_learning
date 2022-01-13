import supersuit
import pettingzoo.mpe.simple_tag_v2 as simple_tag_v2
import gym
import random
import numpy as np

from agent.ppo_agent import PPOAgent

def get_action():
    return np.random.randint(0,5)

env_config = {
    "num_good" : 3,
    "num_adversaries" : 3,
    "num_obstacles" : 2,
    "max_cycles" : 25,
    "continuous_actions" : False
}


if __name__ == "__main__":
    env = simple_tag_v2.env(num_good=env_config["num_good"], num_adversaries=env_config["num_adversaries"], \
        num_obstacles=env_config["num_obstacles"], max_cycles=env_config["max_cycles"], \
            continuous_actions=env_config["continuous_actions"])

    agent = PPOAgent(14, 5)

    for i_eps in range(10000):
        env.reset()
        state = env.last()
        print(state)
        print(np.asarray(state[0]).shape)
        step_cnt = 0

        for _ in range(1000):
            step_cnt += 1
            print('step cnt : {}'.format(step_cnt))
            action = agent.get_action()
            env.step(action)
            next_state, reward, done, info = env.last()
            env.render()
            print('next_state : {}, reward : {}, done : {}, info : {}'.format(next_state, reward, done, info))
