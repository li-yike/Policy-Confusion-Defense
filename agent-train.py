import gym
import numpy as np
import os
import time
import argparse
import random
os.environ['LANG'] = 'en-US'
from sb3_contrib import TRPO
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from stable_baselines3.common.evaluation import evaluate_policy 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str)
    parser.add_argument('--task',type=str,default='HalfCheetah-v3')
    parser.add_argument('--seed',type=int,default=15)
    parser.add_argument('-n',type=int,default=2000000)
    args = parser.parse_args()
    env = gym.make(args.task)
    algo = args.algo
    if args.seed == 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    try:
        if algo == 'ppo':
            model = PPO("MlpPolicy", env, verbose=1,seed=seed)
            model.learn(total_timesteps=args.n, log_interval=4)
        elif algo == 'trpo':
            model = TRPO("MlpPolicy", env, verbose=1,seed=seed)
            model.learn(total_timesteps=args.n, log_interval=4)
        elif algo == 'ddpg':
            model = DDPG("MlpPolicy", env, verbose=1,seed=seed)
            model.learn(total_timesteps=args.n, log_interval=4)
        elif algo == 'sac':
            model = SAC("MlpPolicy", env, verbose=1,seed=seed)
            model.learn(total_timesteps=args.n, log_interval=4)
    
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        pass
    finally:
        # Clean progress bar
        try:
            model_name = "/root/autodl-tmp/leak_agent/"+args.algo + "_" + args.task + "_seed="+str(seed)  
            model.save(model_name)
            model.env.close()
        except EOFError:
            pass

    del model # remove to demonstrate saving and loading
    
def train():
    env = gym.make("HalfCheetah-v3") 
    seed = 30               
    algo = "sac"            
    task = "HalfCheetah"    
    model = PPO("MlpPolicy", env, verbose=1,seed=seed)
    try:
        model.learn(total_timesteps=5000000, log_interval=4)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print("leak look look")
        print(f"After training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        pass
    finally:
        # Clean progress bar
        try:
            model_name = "leak_agent/"+ algo + "_" + task + "_seed="+str(seed)  
            model.save(model_name)
            model.env.close()
        except EOFError:
            pass

    del model # remove to demonstrate saving and loading
    
def train_atari():
    seed = 19
    env = make_atari_env("Breakout-v4", n_envs=4, seed=seed)
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)
    model = SAC("CnnPolicy", env, verbose=1)
    algo = 'sac'
    task = 'Breakout-v4'
    try:
        model.learn(total_timesteps=1e7)
        
    except KeyboardInterrupt:
        # this allows to save the model when interrupting training
        pass
    finally:
        # Clean progress bar
        try:
            model_name = "agent/"+ algo + "_" + task+"_n=1e7" + "_seed="+str(seed)  
            model.save(model_name)
            model.env.close()
        except EOFError:
            pass

    del model # remove to demonstrate saving and loading
    
    
if __name__ == "__main__":
    # train_atari()
    main()
    
   