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


# 定义原始运行的游戏环境
env_name = 'Hopper-v3'
deterministic = True
gym_env = gym.make(env_name)
# 载入智能体
policy = []
print("Model Loading........")
# print("Load Model: agent/trpo_HalfCheetah_seed=34")
# model1 = TRPO.load('/root/agent/trpo_HalfCheetah_seed=34')
# policy.append(model1)
# print("Load Model: agent/ppo_HalfCheetah_seed=61")
# model2 = PPO.load('/root/agent/ppo_HalfCheetah_seed=61')
# policy.append(model2)
# print("Load Model: agent/ddpg_HalfCheetah_seed=34")
# model3 = DDPG.load('/root/agent/ddpg_HalfCheetah_seed=34')
# policy.append(model3)
# print("Load Model: agent/sac_HalfCheetah")
# model4 = SAC.load('/root/agent/sac_HalfCheetah')
# policy.append(model4)

# print("Load Model: agent/trpo_Ant")
# model1 = TRPO.load('/root/agent/trpo_Ant')
# policy.append(model1)
# print("Load Model: agent/ppo_Ant")
# model2 = PPO.load('/root/agent/ppo_Ant')
# policy.append(model2)
# print("Load Model: agent/ddpg_ant")
# model3 = DDPG.load('/root/agent/ddpg_ant')
# policy.append(model3)
# print("Load Model: agent/sac_Ant")
# model4 = SAC.load('/root/agent/sac_Ant')
# policy.append(model4)

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

t0 = time.time()
print("start time t0 = ", t0)

# 定义Swtich Agent的训练环境
env = SwitchEnv(env_name,policy,deterministic,1000,0)

def exec_model():
    #model = PPO.load('/root/autodl-tmp/models/Switch-Agent/PPO-Walker-lambda=0.3/400000.0')
    # model = PPO.load('/root/leak_result_2/trained_models/PPO-Walker-lambda=0.3/400000.0')
    # model = PPO.load('/root/autodl-tmp/leak_models/Switch-Agent/4-sac-Hopper-lambda=0.3/200000.0')######################
    # model = PPO.load('/root/autodl-tmp/leak_models/Switch-Agent/4-sac-Walker2d-lambda=0.3/200000.0')######################
    # 0531
    # model = PPO.load('/root/autodl-tmp/leak_models/Switch-Agent/Time_4-sac-HalfCheetah-lambda=0.3/300000.0')
    # model = PPO.load('/root/autodl-tmp/leak_models/Switch-Agent/Time_PCD_Ant-lambda=0.8/100000.0')
    model = PPO.load('/root/autodl-tmp/leak_models/Switch-Agent/Time_PCD_Hopper-lambda=0.8/100000.0')
    
    obs = env.reset()
    n_trajs = 10 ### 100 Time实验10
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
            action, _ = model.predict(obs, deterministic=True) # 预测动作
            # 修改了 common/policies 里面的 _predict 方法 以及 新增了 common/base_class 里面的 predict_prob 方法
            env.action_probs = model.predict_prob()[0] # 选择各个动作的概率
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
    t1 = time.time()
    print("start time t1 = ", t1)
    print("rollout time t = ", t1-t0) #用户使用时间开销 实验
    print("Reward Avg: {} +/- {}".format(np.mean(total_rewards),np.std(total_rewards)))
    print("Length Avg: {} +/- {}".format(np.mean(lengths),np.std(lengths)))
    # # 保存运行轨迹
    # filename = '/root/leak_result_2/traj/ppo-defender-Walker2d-2trpo+ppo+ddpg-100-lambda=0.3-400k.h5'
    # filename = '/root/autodl-tmp/leak_models/Switch-Agent-trajs/4-sac-Hopper-lambda=0.3-n_trajs=100-iter=200000.h5'
    # filename = '/root/autodl-tmp/leak_models/Switch-Agent-trajs/4-sac-Walker2d-lambda=0.3-n_trajs=100-iter=200000.h5'
    # 0531
    filename = '/root/autodl-tmp/leak_models/Switch-Agent-trajs/Time_Hopper_PCD.h5'
    with h5py.File(filename, 'w') as f:
        f.create_dataset('obs_B_T_Do', data=observation)
        f.create_dataset('a_B_T_Da', data=actions)
        f.create_dataset('r_B_T', data=rewards)
        f.create_dataset('len_B', data=lengths)
    
exec_model()
