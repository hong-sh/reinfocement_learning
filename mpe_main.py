import supersuit
import pettingzoo.mpe.simple_tag_v2 as simple_tag_v2
import gym
import random
import numpy as np

from agent.ppo_agent import PPOAgent

def get_action():
    return np.random.randint(0,5)

env_config = {
    "num_good" : 2,
    "num_adversaries" : 3,
    "num_obstacles" : 2,
    "max_cycles" : 100,
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
                # print('next_state : {}, reward : {}, done : {}, info : {}'.format(next_state, reward, done, info))
                action = 0
                if agent_idx < env_config["num_adversaries"]:
                    action, action_prob = adversary_agent.get_action(next_state)
                    adversary_agent.save_xp((prev_state[agent_idx], next_state, action, action_prob[action].item(), reward, done))
                    prev_state[agent_idx] = next_state
                    sum_reward += reward
                elif agent_idx >= env_config["num_adversaries"]:
                    action = get_action()

                env.step(action)
                # env.render()
            step_cnt += 1
        adversary_agent.train()
        print('{} eps total reward : {}'.format(i_eps, sum_reward))
