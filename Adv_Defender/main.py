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
# env_name = 'Hopper-v3'
deterministic = True
gym_env = gym.make(env_name)
# 载入智能体
policy = []

print("Model Loading........")
print("Load Model: agent/trpo_hopper_1")
model1 = TRPO.load('/root/agent/trpo_hopper_1')
policy.append(model1)
print("Load Model: agent/ppo_Hopper-v3_1")
model2 = PPO.load('/root/agent/ppo_Hopper-v3_1')
policy.append(model2)
print("Load Model: agent/ddpg_hopper_seed=13")
model3 = DDPG.load('/root/agent/ddpg_hopper_seed=50')
policy.append(model3)
print("Load Model: agent/sac_Hopper")
model4 = SAC.load('/root/agent/sac_Hopper')
policy.append(model4)

# Define SwtichEnv
max_steps = 1000
Mj_traj = 5
env = SwitchEnv(env_name,policy,deterministic,max_steps,Mj_traj)

# Test SwtichEnv
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
    # Train Switcher Agent
    models_dir = '/root/autodl-tmp/modify_models/modify_hopper_m'
    logdir = '/root/tf-logs/modify_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=logdir)
    # Save model
    TIMESTEPS = 1e5 

    try:
        for i in range(1,31):
            model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name='modify_hopper_m')
            # env.model = model # callback
            model.save(f'{models_dir}/{TIMESTEPS*i}')
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        pass
    finally:
        # Clean progress bar
        try:
            model_name = os.path.join(models_dir, "modify_hopper_m")
            model.save(model_name)
        except EOFError:
            pass
    
    # Run the trained switcher agent
    
    
if __name__ == "__main__":
    
    # If the environment don't follow the interface, an error will be thrown
    train_agent()
    # exec_model()
