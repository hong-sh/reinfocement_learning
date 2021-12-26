import supersuit
import pettingzoo.mpe.simple_v2 as simple_v2
import random
import numpy as np

def get_action():
    return np.random.randint(0,5)

if __name__ == "__main__":
    env = simple_v2.env(max_cycles=1000, continuous_actions=False)

    env.reset()
    for _ in range(1000):
        action = get_action()
        env.step(action)
        next_state, reward, done, info = env.last()
        print('next_state : {}, reward : {}, done : {}, info : {}'.format(next_state, reward, done, info))
        env.render()
