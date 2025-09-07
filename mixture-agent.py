import gym
import numpy as np
import os
import time
import h5py
import argparse
import random
os.environ['LANG'] = 'en-US'
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
# python3 mixture-test.py agent/trpo_hopper_1.zip agent/trpo_hopper_4.zip --task Hopper-v3 --algo trpo --interval 300 > traj_mixture-model_num=300.log
def model_predict(model,obs,deterministic=False):
    action,_states = model.predict(obs,deterministic=deterministic)
    return action

def choose_action(obs,deterministic=False):
    actions = np.zeros(env.action_space.shape)
    for i in range(model_num):
        action = model_predict(models[i],obs,deterministic)
        actions += action
    print(actions/model_num)
    return actions / model_num

parser = argparse.ArgumentParser()
parser.add_argument('policy', type=str,nargs='+')
parser.add_argument('--task',type=str,default="Ant-v3")
parser.add_argument('--mixture',type=int,default=4)
# 0: switch agent every 'interval' seconds
# 1: switch agent every 'interval' steps
# 2: switch agent every 'interval' steps, but the first agent is random
# 3: randomly switch agent every step
# 4: randomly choose an 'interval', switch agent every 'interval' steps
parser.add_argument('--interval',type=int,default=1)
parser.add_argument('--trajs_num',type=int,default=100)
parser.add_argument('--savepath',type=str,default="/root/leak_result")
parser.add_argument('--max_trajs_len', type=int, default=1000)
parser.add_argument('--deterministic',type=bool,default=True)
args = parser.parse_args()
deterministic = args.deterministic
savepath = args.savepath
args = parser.parse_args()
env = gym.make(args.task)
print("Model Loading........")
models = []
algo = ""
for policystr in args.policy:
    print("Loading ",policystr)
    if 'trpo' in policystr.split("_")[0]:
        algo += 'trpo'
        models.append(TRPO.load(policystr))
    elif 'ppo' in policystr.split("_")[0]:
        algo += 'ppo'
        models.append(PPO.load(policystr))
    elif 'sac' in policystr.split("_")[0]:
        algo += 'sac'
        models.append(SAC.load(policystr))
    elif 'ddpg' in policystr.split("_")[0]:
        algo += 'ddpg'
        models.append(DDPG.load(policystr))
    algo += '-'
print("Model Load Complete!")
print("Model num: ",len(models))
obs = env.reset()
model_num = len(models)
observation = np.zeros([args.trajs_num,args.max_trajs_len,env.observation_space.shape[0]],dtype=float)
actions = np.zeros([args.trajs_num,args.max_trajs_len,env.action_space.shape[0]],dtype=float)
rewards = np.zeros([args.trajs_num,args.max_trajs_len,1],dtype=float)
start_time = time.time()
interval = args.interval
lengths = []


episode_reward = []
for i in range(args.trajs_num):
    total_reward = 0
    step = 0
    if args.mixture == 4:
        interval = random.randint(1,args.max_trajs_len) 
        print("Eposide {} Interval: {}".format(i,interval))
    while True:
        if args.mixture == 0:
            model_select = ((time.time() - start_time) // args.interval) % model_num
        elif args.mixture == 1:
            if interval == 1:
                model_select = random.randint(0,model_num-1)
            else:
                model_select = step // interval % model_num # Switch agent every 'interval' steps
        elif args.mixture == 2 or args.mixture == 4:
            if step == 0:
                # print(model_num)
                model_select = random.randint(0,model_num-1)
                # print(model_select)
            else:
                model_select = step // interval % model_num
        elif args.mixture == 3:
            model_select = random.randint(0,3) #4 agent

        observation[i][step] = obs
        if args.mixture != 5:
            print("Using Agent ",model_select)
            action = model_predict(models[int(model_select)],obs,deterministic)
        else:
            action = choose_action(obs,deterministic)
        obs,reward,done,info = env.step(action)
    #	env.render()
        actions[i][step] = action
        rewards[i][step] = reward
        total_reward += reward
        step += 1
        if done or step > args.max_trajs_len:
            print("Eposide {} Reward: {}".format(i,total_reward))
            episode_reward.append(total_reward)
            obs = env.reset()
            lengths.append(step)
            break

print("Leak look look episode_reward =", episode_reward)
print("Reward Avg: {} +/- {}".format(rewards.sum(axis=1).mean(),rewards.sum(axis=1).std()))
print("Length Avg: {} +/- {}".format(np.mean(lengths),np.std(lengths)))
# save trajectory
filename = "leak"+algo + args.task + "_step-interval="+ str(interval) + "-model-num="+str(model_num)+"-mixture.h5"
filename = os.path.join(savepath,filename)
with h5py.File(filename, 'w') as f:
    f.create_dataset('obs_B_T_Do', data=observation)
    f.create_dataset('a_B_T_Da', data=actions)
    f.create_dataset('r_B_T', data=rewards)
    f.create_dataset('len_B', data=lengths)
