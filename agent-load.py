import gym
import numpy as np
import os
import time
import argparse
import random
import h5py
os.environ['LANG'] = 'en-US'
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'Hopper-v3'
deterministic = True
env = gym.make(env_name)
# Load agents
policy = []
print("Model Loading........")
print("Load Model: agent/ppo_Hopper-v3_1")
model1 = PPO.load('/root/agent/ppo_Hopper-v3_1')
policy.append(model1)
print("Load Model: agent/trpo_hopper_4")
model2 = TRPO.load('/root/agent/trpo_hopper_4')
policy.append(model2)
print("Load Model: agent/trpo_hopper_1")
model3 = TRPO.load('/root/agent/trpo_hopper_1')
policy.append(model3)
print("Load Model: agent/ddpg_hopper_seed=1")
model4 = DDPG.load('/root/agent/ddpg_hopper_seed=13')#'/root/autodl-tmp/agent/ddpg_hopper_seed=13'
policy.append(model4)
obs = env.reset()
trajs_num = 100
max_trajs_len = 1000

def agent_load():
    for model in policy:
        mean_reward, std_reward = evaluate_policy(model, env)
        print(f"Policy: { model } Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

def stack_action():
    start_time = time.time()
    lengths = []
    observation = np.empty([trajs_num,max_trajs_len,env.observation_space.shape[0]],dtype=float)
    mix_actions = np.empty([trajs_num,max_trajs_len,env.action_space.shape[0]],dtype=float)
    rewards = np.empty([trajs_num,max_trajs_len],dtype=float)
    for i in range(trajs_num):
        step = 0
        total_reward = 0
        while True:
            observation[i][step] = obs
            actions = np.zeros(env.action_space.shape)
            action,_states = model1.predict(obs,deterministic=True)
            actions += action
            action,_states = model2.predict(obs,deterministic=True)
            actions += action
            action,_states = model3.predict(obs,deterministic=True)
            actions += action
            action,_states = model4.predict(obs,deterministic=True)
            actions += action
            action = actions / 4
            # print("Stack action: ",action)
            obs,reward,done,info = env.step(action)
        #   	env.render()
            mix_actions[i][step] = action
            rewards[i][step] = reward
            total_reward += reward
            step = step + 1
            if done or step > max_trajs_len:
                print("Eposide {} Reward: {} Length: {}".format(i,total_reward,step))
                obs = env.reset()
                lengths.append(step)
                break
    print("Reward Avg: {} +/- {}".format(rewards.sum(axis=1).mean(),rewards.sum(axis=1).std()))
    print("Length Avg: {} +/- {}".format(np.mean(lengths),np.std(lengths)))
    # save data
    savepath = "/root/autodl-tmp/traj"
    filename = "ppo_hopper.h5"
    filename = os.path.join(savepath,filename)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('obs_B_T_Do', data=observation)
        f.create_dataset('a_B_T_Da', data=mix_actions)
        f.create_dataset('r_B_T', data=rewards)
        f.create_dataset('len_B', data=lengths)
    f.close()
    
def generate_traj():
    observations = np.empty([trajs_num,max_trajs_len,env.observation_space.shape[0]],dtype=float)
    actions = np.empty([trajs_num,max_trajs_len,env.action_space.shape[0]],dtype=float)
    rewards = np.empty([trajs_num,max_trajs_len],dtype=float)
    lengths = []
    for i in range(trajs_num):
        obs = env.reset()
        step = 0
        total_reward = 0
        while True:
            observations[i][step] = obs
            action,_states = model1.predict(obs,deterministic=True)
            obs,reward,done,info = env.step(action)
        #   	env.render()
            actions[i][step] = action
            rewards[i][step] = reward
            total_reward += reward
            step = step + 1
            if done or step > max_trajs_len:
                print("Eposide {} Reward: {} Length: {}".format(i,total_reward,step))
                lengths.append(step)
                break
    print("Reward Avg: {} +/- {}".format(rewards.sum(axis=1).mean(),rewards.sum(axis=1).std()))
    print("Length Avg: {} +/- {}".format(np.mean(lengths),np.std(lengths)))
    # Save Trajectory
    savepath = "/root/autodl-tmp/traj"
    filename = "ppo_hopper.h5"
    filename = os.path.join(savepath,filename)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('obs_B_T_Do', data=observations)
        f.create_dataset('a_B_T_Da', data=actions)
        f.create_dataset('r_B_T', data=rewards)
        f.create_dataset('len_B', data=lengths)
    f.close()

generate_traj()
# agent_load()