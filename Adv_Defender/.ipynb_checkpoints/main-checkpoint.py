from SwitchEnv.switchEnv import SwitchEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import SAC
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
# print("Model Loading........")
# print("Load Model: agent/sac_Hopper")
# model1 = SAC.load('/root/agent/sac_Hopper')
# policy.append(model1)
# print("Load Model: agent/sac_Hopper-v3_seed=13")
# model2 = SAC.load('/root/agent/sac_Walker2d-v3_seed=13')
# policy.append(model2)
# print("Load Model: agent/sac_Walker2d-v3_seed=28")
# model3 = SAC.load('/root/agent/sac_Walker2d-v3_seed=28')
# policy.append(model3)
# print("Load Model: agent/sac_Walker2d-v3_seed=45")
# model4 = SAC.load('/root/agent/sac_Walker2d-v3_seed=45')
# policy.append(model4)

# print("Model Loading........")
# print("Load Model: agent/sac_Walker2d")
# model1 = SAC.load('/root/agent/sac_Walker2d')
# policy.append(model1)
# print("Load Model: agent/sac_Walker2d-v3_seed=13")
# model2 = SAC.load('/root/agent/sac_Walker2d-v3_seed=13')
# policy.append(model2)
# print("Load Model: agent/sac_Walker2d-v3_seed=28")
# model3 = SAC.load('/root/agent/sac_Walker2d-v3_seed=28')
# policy.append(model3)
# print("Load Model: agent/sac_Walker2d-v3_seed=45")
# model4 = SAC.load('/root/agent/sac_Walker2d-v3_seed=45')
# policy.append(model4)

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
print("Load Model: agent/sac_Hopper")
model4 = SAC.load('/root/agent/sac_Hopper')
policy.append(model4)

# 定义Swtich Agent的训练环境
max_steps = 1000
Mj_traj = 5
env = SwitchEnv(env_name,policy,deterministic,max_steps,Mj_traj)

# 测试SwtichEnv是否正常
def test_env():
    obs = env.reset()
    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())
    n_steps = 10
    for step in range(n_steps):
        choose_policy = random.randint(0,1)
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(choose_policy)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break

def train_agent():
    # wrap it
    # env = make_vec_env(env, n_envs=1)
    # 使用PPO训练Switch Agent
    # models_dir = '/root/autodl-tmp/leak_models/Switch-Agent/4-sac-Hopper-lambda=0.3'#############
    # models_dir = '/root/autodl-tmp/leak_models/Switch-Agent/4sac-Walker2d-lambda=0.05'#############
    models_dir = '/root/autodl-tmp/leak_models/Switch-Agent/Time_PCD_Hopper-lambda=0.8'
    logdir = '/root/tf-logs/Switch-Agent'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=logdir)
    # 保存Switch Agent
    TIMESTEPS = 1e5 #################

    try:
        for i in range(1,31):
            # model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name='sac-4-opper-lambda=0.3-traj=5')#############
            # model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name='4sac-Walker2d-lambda=0.05-traj=5')
            model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name='Time_PCD_Hopper-lambda=0.8-traj=5')
            # env.model = model # callback
            model.save(f'{models_dir}/{TIMESTEPS*i}')
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        pass
    finally:
        # Clean progress bar
        try:
            # model_name = os.path.join(models_dir, "sac-4-Hopper_Switch-Agent-3M")#############
            model_name = os.path.join(models_dir, "Time_PCD_Hopper")
            model.save(model_name)
        except EOFError:
            pass
    
    # 运行训练好的Switch Agent 
    
    
if __name__ == "__main__":
    
    t0 = time.time()
    print("start time t0 = ", t0)
    
    # If the environment don't follow the interface, an error will be thrown
    train_agent()
    # exec_model()
    
    t1 = time.time()
    print("end time t1 = ", t1)
