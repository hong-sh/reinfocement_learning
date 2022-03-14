from re import A
import supersuit
import pettingzoo.mpe.simple_tag_v2 as simple_tag_v2
import gym
import random
import numpy as np
from time import time_ns
from tensorboardX import SummaryWriter

from agent.ppo_agent import PPOAgent

def get_action():
    return np.random.randint(0,5)

env_config = {
    "num_good" : 2,
    "num_adversaries" : 3,
    "num_obstacles" : 2,
    "max_cycles" : 1000,
    "continuous_actions" : False
}

if __name__ == "__main__":
    env = simple_tag_v2.env(num_good=env_config["num_good"], num_adversaries=env_config["num_adversaries"], \
        num_obstacles=env_config["num_obstacles"], max_cycles=env_config["max_cycles"], \
            continuous_actions=env_config["continuous_actions"])
    
    sum_of_agents = env_config["num_good"] + env_config["num_adversaries"]
    adversary_observation = 4 + (env_config["num_obstacles"] * 2) + (env_config["num_good"] + env_config["num_adversaries"]-1) * 2 + env_config["num_good"] * 2
    good_observation = 4 + (env_config["num_obstacles"] * 2) + (env_config["num_good"] + env_config["num_adversaries"]-1) * 2 + (env_config["num_good"]-1) * 2

    adversary_agent = PPOAgent(adversary_observation , 5)
    good_agent = PPOAgent(good_observation, 5)

    summary_writer = SummaryWriter('logs/mpe_main_' + str(time_ns()))

    for i_eps in range(10000):
        env.reset()
        prev_state = [np.zeros(adversary_observation) for _ in range(env_config["num_adversaries"])]
        step_cnt = 0
        sum_reward = 0

        while step_cnt < env_config["max_cycles"] * sum_of_agents:
            agent_idx = step_cnt % sum_of_agents
            next_state, reward, done, info = env.last()
            if not done:
                # print('step cnt : {}'.format(step_cnt))
                action = 0
                if agent_idx < env_config["num_adversaries"]:
                    action, action_prob = adversary_agent.get_action(next_state)
                    print('agent_idx : {}, prev_state : {}'.format(agent_idx, prev_state[agent_idx])                     )
                    print('action : {}, next_state : {}, reward : {}, done : {}, info : {}'.format(action, next_state, reward, done, info))
                    adversary_agent.save_xp((prev_state[agent_idx], next_state, action, action_prob[action].item(), reward, done))
                    # TODO fix wrong cycles 
                    prev_state[agent_idx] = next_state
                    sum_reward += reward
                elif agent_idx >= env_config["num_adversaries"]:
                    action = get_action()

                env.step(action)
                # env.render()
            step_cnt += 1
        pi_loss, value_loss, td_error =  adversary_agent.train()
        summary_writer.add_scalar('Loss/pi_loss', pi_loss, i_eps)
        summary_writer.add_scalar('Loss/value_loss', value_loss, i_eps)
        summary_writer.add_scalar('Loss/td_error', td_error, i_eps)
        summary_writer.add_scalar('Episode reward', sum_reward, i_eps)
        print('{} eps total reward : {}'.format(i_eps, sum_reward))

