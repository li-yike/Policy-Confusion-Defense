import tensorflow.compat.v1 as tf
import h5py
import numpy as np
import gym
import time
import os
import pickle
tf.disable_eager_execution()


def load_task_data(trajs_filename,limit_trajs):
    task_data = {}
    with h5py.File(trajs_filename,'r') as f:
        obs = f['obs_B_T_Do'][:limit_trajs,...][...]
        a = f['a_B_T_Da'][:limit_trajs,...][...]
        r = f['r_B_T'][:limit_trajs,...][...]
        len_B = f['len_B'][:limit_trajs,...][...]
    f.close()
    task_data['observations'] = obs.reshape(-1,obs.shape[-1])
    task_data['actions'] = a.reshape(-1,a.shape[-1])
    task_data['returns'] = r
    return task_data

def load_pkl_data(trajs_filename):
    with open(trajs_filename,'rb') as f:
        task_data = pickle.load(f)
    return task_data

for time in range(1,53):
    expert_data_file = '/root/autodl-tmp/traj/PPO-BC/ppo-defender-Hopper-2trpo+ppo+ddpg-100-defend-' + str(time) + '.h5'
    limit_trajs = 100
    task_data = load_task_data(expert_data_file,limit_trajs) #"data/" + data_file + ".pkl")
    print(f"Episode {time} Expert Reward: {task_data['returns'].sum(axis=1).mean()}")
    #create a Feedforward network useing Keras