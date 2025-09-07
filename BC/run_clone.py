import pickle
import tensorflow.compat.v1 as tf
import argparse
import h5py
import numpy as np
# from BC import tf_util
import tf_util
import gym
# from BC.load_policy import *
import load_policy
import time
import os
tf.disable_eager_execution()
from keras.models import Sequential, load_model
from keras.layers import Dense 

class Behavior_clone():
    def __init__(self,env_name,savepath,sample_num):
        self.env = gym.make(env_name)
        self.model = Sequential()
        self.model.add(Dense(96, activation = "relu", input_shape = (self.env.observation_space.shape[0],)))
        self.model.add(Dense(96, activation = "relu"))
        self.model.add(Dense(96, activation = "relu"))
        self.model.add(Dense(self.env.action_space.shape[0], activation = "linear"))
        self.model_path = savepath
        self.sample_num = sample_num

    def train(self,obs_data,act_data,epoch,train_time):
        obs_data = obs_data.reshape(-1,obs_data.shape[-1])
        act_data = act_data.reshape(-1,act_data.shape[-1])
        self.model.compile(loss = "mean_squared_error", optimizer = "adam", metrics=["accuracy"])
        self.model.fit(obs_data, act_data, batch_size = 64, epochs = epoch, verbose = 1)
        model_name = 'train-time='+str(train_time)+ '_cloned_model.h5'
        model_name = os.path.join(self.model_path,model_name)
        print(model_name)
        self.model.save(model_name)
        print('Model Save: ',model_name)

    def exec(self):
        with tf.Session():
            tf_util.initialize()

            max_steps = self.env.spec.max_episode_steps

            returns = []
            for i in range(self.sample_num):
                print('iter', i)
                obs = self.env.reset()

                done = False
                totalr = 0.
                steps = 0

                cloned_model = self.model
                while not done:
                    obs=np.array(obs)
                    action = cloned_model.predict(obs[None,:], batch_size = 64, verbose = 0)
                    time.sleep(.005)
                    obs, r, done, _ = self.env.step(action)
                    totalr += r
                    steps += 1

                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)
            
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
        return np.mean(returns)


def load_task_data(trajs_filename,limit_trajs):
    task_data = {}
    with h5py.File(trajs_filename,'r') as f:
        obs = f['obs_B_T_Do'][:limit_trajs]
        a = f['a_B_T_Da'][:limit_trajs]
        r = f['r_B_T'][:limit_trajs]
        # len_B = f['len_B'][:limit_trajs]
    f.close()
    task_data['observations'] = obs.reshape(-1,obs.shape[-1])
    task_data['actions'] = a.reshape(-1,a.shape[-1])
    task_data['returns'] = r
    return task_data


def run_clone(expert_name, expert_data_file,savepath,limit_trajs=4, epoch=30,num_rollouts = 100,render = False): 
    #expert_name: the gym expert policy name
    #render: True to render
    #num_rollouts
    
    savepath1="/root/leak_result/BC_learned/data/"     ################
    task_data = load_task_data(expert_data_file,limit_trajs) #"data/" + data_file + ".pkl")
    algo = expert_data_file.split('/')[-1].split("_")[0]
    obs_data = np.array(task_data["observations"])
    act_data = np.array(task_data["actions"])
    print("Expert Reward: ",task_data['returns'].sum(axis=1).mean())
    #create a Feedforward network useing Keras

    model = Sequential()
    model.add(Dense(96, activation = "relu", input_shape = (obs_data.shape[1],)))
    model.add(Dense(96, activation = "relu"))
    model.add(Dense(96, activation = "relu"))
    model.add(Dense(act_data.shape[1], activation = "linear"))

    model.compile(loss = "mean_squared_error", optimizer = "adam", metrics=["accuracy"])
    model.fit(obs_data, act_data, batch_size = 64, epochs = epoch, verbose = 1)
    
    model_name = 'models/' + "Switch" + expert_name+ "_algo="+algo+"_epoch="+ str(epoch) +"_limit-trajs="+ str(limit_trajs) + '_cloned_model.h5'###_cloned_model_2.h5
    print(model_name)
    model_name = os.path.join(savepath,model_name)
    print(model_name)
    model.save(model_name)
    print('Model Save: ',model_name)
    with tf.Session():
        tf_util.initialize()

        env = gym.make(expert_name)
        max_steps = env.spec.max_episode_steps

        returns = []
        cloned_observations = []
        cloned_actions = []
        cloned_length = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()

            done = False
            totalr = 0.
            steps = 0

            cloned_model = load_model(model_name)
            while not done:
                obs=np.array(obs)
                action = cloned_model.predict(obs[None,:], batch_size = 64, verbose = 0)
                cloned_observations.append(obs)
                cloned_actions.append(action)
                time.sleep(.005)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1

                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
            cloned_length.append(steps)

        data = {'observations': np.array(cloned_observations),
                       'actions': np.array(cloned_actions),
                       'returns': np.array(returns),
                       'len':np.array(cloned_length)}
        
        output_file_name = "Switch" + expert_name + '_' + str(num_rollouts) + "_algo="+algo+"_epoch="+ str(epoch)+ "_limit-trajs="+ str(limit_trajs) + '_clone.pkl' #################clone_2
        output_file_name = os.path.join(savepath1,output_file_name)
        
        with open(output_file_name, 'wb') as f:
            pickle.dump(data, f)
        print("Trajs Save: ",output_file_name)
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        
        return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="Hopper-v3")
    parser.add_argument('--expert_data',type=str,required=True)
    parser.add_argument('--trajs_num', type=int, default=100)
    parser.add_argument('--limit_trajs',type=int,default=100)
    parser.add_argument('--savepath',type=str)
    parser.add_argument('--epoch',type=int,default=30) 
    args = parser.parse_args()
    print(args)
    expert_filename = args.expert_data
    env_name = args.env_name
    trajs_num = args.trajs_num
    epoch = args.epoch
    returns = run_clone(env_name,expert_filename,args.savepath,args.limit_trajs,epoch,trajs_num)

if __name__ == "__main__":
    main()
