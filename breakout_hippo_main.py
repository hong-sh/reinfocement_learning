import gym
import os
from time import time
import numpy as np

from agent.hippo_cnn_agent import HiPPOCNNAgent
from tensorboardX import SummaryWriter
from utils.atari_wrappers import make_env


time_str = str(time())
log_dir = 'logs/breakout_hippo_main_' + time_str
# model_dir = 'save_model/mpe_main_' + time_str
# os.mkdir(model_dir)

T_horizon = 256

def get_stick_loc(state):
    stick_color= state[191, 152]
    stick_bound = state[191, 8:152]
    stick_center = np.where(stick_bound==stick_color)[0][0] + 8
    left = stick_center / 144
    return left

def calculate_low_level_reward(next_state, h_action_logits):
    l_reward = 0.0
    left = get_stick_loc(next_state)
    if  h_action_logits[0]  * 1.2 >= left and h_action_logits[1] * 0.8 <= left:
        l_reward = 0.01
    else:
        l_reward = -0.01 * abs(h_action_logits[0]  - left)

    return l_reward

if __name__ == "__main__":
    env = gym.make('BreakoutNoFrameskip-v4')
    env = make_env(env)
    env.seed(42)
    hippo_agent = HiPPOCNNAgent(num_inputs=4, high_level_action_space=2, low_level_action_space=4,\
        min_period=2, max_period=4, random_period=True)

    summary_writer = SummaryWriter(log_dir)

    score = 0.0
    global_step = 0
    for i in range(10000000000):
        hippo_agent.reset_count()
        state = env.reset()
        state = np.asarray(state)
        state = state.transpose((2, 0, 1))
        done = False
        local_step = 0

        while not done:
            for t in range(T_horizon):
                h_action, h_action_probs, h_action_logits, l_action, l_action_probs = hippo_agent.get_action(state)

                # summary_writer.add_scalar('Episode/action', action, global_step)
                next_state, reward, done, info = env.step(l_action)

                h_reward = reward + 0.01
                l_reward = reward = calculate_low_level_reward(env.metadata["original_frame"], h_action_logits)
                next_state = np.asarray(next_state)
                next_state = next_state.transpose((2, 0, 1))
                # print('next_state : {}, action : {}, reward : {}, done : {}, info : {}'.format(next_state, action, reward, done, info))

                hippo_agent.save_xp((state, next_state, h_action, h_action_probs, h_action_logits, h_reward, l_action, l_action_probs, l_reward, done))
                
                state = next_state
                score += reward
                local_step += 1

                if done:
                    break
            
            h_pi_loss, h_value_loss, h_dist_entropy, h_approx_kl, h_approx_ent, h_clipfrac, \
                l_pi_loss, l_value_loss, l_dist_entropy, l_approx_kl, l_approx_ent, l_clipfrac = hippo_agent.train()

            summary_writer.add_scalar('HLoss/pi_loss', h_pi_loss, global_step)
            summary_writer.add_scalar('HLoss/value_loss', h_value_loss, global_step)
            summary_writer.add_scalar('HLoss/dist_entropy', h_dist_entropy, global_step)
            summary_writer.add_scalar('HLoss/approx_kl', h_approx_kl, global_step)
            summary_writer.add_scalar('HLoss/approx_ent', h_approx_ent, global_step)
            summary_writer.add_scalar('HLoss/clipfrac', h_clipfrac, global_step)

            summary_writer.add_scalar('LLoss/pi_loss', l_pi_loss, global_step)
            summary_writer.add_scalar('LLoss/value_loss', l_value_loss, global_step)
            summary_writer.add_scalar('LLoss/dist_entropy', l_dist_entropy, global_step)
            summary_writer.add_scalar('LLoss/approx_kl', l_approx_kl, global_step)
            summary_writer.add_scalar('LLoss/approx_ent', l_approx_ent, global_step)
            summary_writer.add_scalar('LLoss/clipfrac', l_clipfrac, global_step)
            global_step += 1

            # print('pi_loss : {}, value_loss : {}, dist_entropy : {}'.format(pi_loss, value_loss, dist_entropy))
        # if i % 10 == 0:
        summary_writer.add_scalar('Episode/score', score, i)
        summary_writer.add_scalar('Episode/game_len', local_step, i)
        # print("{} episode avg score : {:.1f}".format(i+1, score/10))
        score = 0.0

        env.close()





