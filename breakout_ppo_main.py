import gym
import os
from time import time
import numpy as np

from agent.ppo_cnn_agent import PPOCNNAgent
from tensorboardX import SummaryWriter
from utils.atari_wrappers import make_env


time_str = str(time())
log_dir = 'logs/breakout_main_' + time_str
# model_dir = 'save_model/mpe_main_' + time_str
# os.mkdir(model_dir)

T_horizon = 256

if __name__ == "__main__":
    env = gym.make('BreakoutNoFrameskip-v4')
    env = make_env(env)
    env.seed(42)
    agent = PPOCNNAgent(4, action_space=4)

    summary_writer = SummaryWriter(log_dir)

    score = 0.0
    global_step = 0
    for i in range(100000):
        state = env.reset()
        state = np.asarray(state)
        state = state.transpose((2, 0, 1))
        done = False
        local_step = 0

        while not done:
            for t in range(T_horizon):
                action, action_probs = agent.get_action(state)
                # summary_writer.add_scalar('Episode/action', action, global_step)
                next_state, reward, done, info = env.step(action)
                reward += 0.02
                next_state = np.asarray(next_state)
                next_state = next_state.transpose((2, 0, 1))
                # print('next_state : {}, action : {}, reward : {}, done : {}, info : {}'.format(next_state, action, reward, done, info))

                agent.save_xp((state, next_state, action, action_probs, reward, done))

                state = next_state
                score += reward
                local_step += 1

                if done:
                    break
            
            pi_loss, value_loss, dist_entropy, approx_kl, approx_ent, clipfrac = agent.train()
            summary_writer.add_scalar('Loss/pi_loss', pi_loss, global_step)
            summary_writer.add_scalar('Loss/value_loss', value_loss, global_step)
            summary_writer.add_scalar('Loss/dist_entropy', dist_entropy, global_step)
            summary_writer.add_scalar('Loss/approx_kl', approx_kl, global_step)
            summary_writer.add_scalar('Loss/approx_ent', approx_ent, global_step)
            summary_writer.add_scalar('Loss/clipfrac', clipfrac, global_step)
            global_step += 1

            # print('pi_loss : {}, value_loss : {}, dist_entropy : {}'.format(pi_loss, value_loss, dist_entropy))
        # if i % 10 == 0:
        summary_writer.add_scalar('Episode/score', score, i)
        summary_writer.add_scalar('Episode/game_len', local_step, i)
        # print("{} episode avg score : {:.1f}".format(i+1, score/10))
        score = 0.0

        env.close()





