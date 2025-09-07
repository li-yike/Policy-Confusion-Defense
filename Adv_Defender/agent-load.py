from SwitchEnv.switchEnv import SwitchEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3.common.policies import obs_as_tensor
from sb3_contrib import TRPO
import gym
import random
import numpy as np
import os
import h5py
import time


# Define the original game environment
env_name = 'Hopper-v3'
deterministic = True
gym_env = gym.make(env_name)
# Load agents
policy = []
print("Model Loading........")

print("Load Model: agent/trpo_hopper_1")
model1 = TRPO.load('/root/agent/trpo_hopper_1')
policy.append(model1)
print("Load Model: agent/ppo_Hopper-v3_1")
model2 = PPO.load('/root/agent/ppo_Hopper-v3_1')
policy.append(model2)
print("Load Model: agent/ddpg_hopper_seed=13")
model3 = DDPG.load('/root/agent/ddpg_hopper_seed=13')
policy.append(model3)
print("Load Model: agent/sac_Hopper-v3_seed=29")
model4 = SAC.load('/root/agent/sac_Hopper-v3_seed=29')
policy.append(model4)

# Define the switching environment
env = SwitchEnv(env_name,policy,deterministic,1000,0)

def exec_model():
    model = PPO.load('/root/autodl-tmp/leak_models/Switch-Agent/Time_PCD_Hopper-lambda=0.8/100000.0')
    
    obs = env.reset()
    n_trajs = 10 
    max_steps = 1000
    observation = np.empty([n_trajs,max_steps,gym_env.observation_space.shape[0]],dtype=float)
    actions = np.empty([n_trajs,max_steps,gym_env.action_space.shape[0]],dtype=float)
    rewards = np.empty([n_trajs,max_steps],dtype=float)
    lengths = []
    total_rewards = []
    for traj in range(n_trajs):
        step = 0
        total_reward = 0
        while True:
            observation[traj][step] = obs 
            action, _ = model.predict(obs, deterministic=True) # Predict action
            env.action_probs = model.predict_prob()[0] 
            # print("Agent: ",action)
            obs, reward, done, info = env.step(action)
            rewards[traj][step] = reward
            total_reward += reward
            step += 1
            if done or step >= max_steps:
                print("Eposide {} Reward: {}".format(traj,total_reward))
                lengths.append(step)
                total_rewards.append(total_reward)
                actions[traj] = env.actions
                obs = env.reset()
                break

    print("Reward Avg: {} +/- {}".format(np.mean(total_rewards),np.std(total_rewards)))
    print("Length Avg: {} +/- {}".format(np.mean(lengths),np.std(lengths)))
    # Save trajectories
    filename = '/root/autodl-tmp/leak_models/Switch-Agent-trajs/Time_Hopper_PCD.h5'
    with h5py.File(filename, 'w') as f:
        f.create_dataset('obs_B_T_Do', data=observation)
        f.create_dataset('a_B_T_Da', data=actions)
        f.create_dataset('r_B_T', data=rewards)
        f.create_dataset('len_B', data=lengths)
    
exec_model()
