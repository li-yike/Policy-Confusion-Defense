import h5py
import argparse
import numpy as np
from dtw import dtw
from numpy.linalg import norm
from policyopt import util
import os,os.path,shutil
import json

# TODO:
# 1. read trajs(obs,action)
# 2. concate obs,action
# 3. compute DTW distance

def readtrajs(trajs_filename):
    with h5py.File(trajs_filename,'r') as f:
        obs = f['obs_B_T_Do'][...][...]
        a = f['a_B_T_Da'][...][...]
        r = f['r_B_T'][...][...]
        len_B = f['len_B'][...][...]
    f.close()
    return obs,a,r,len_B

def load_trained_policy_and_mdp(env_name, policy_state_str):
    import gym
    import policyopt
    from policyopt import nn, rl
    from environments import rlgymenv

    # Load the saved state
    policy_file, policy_key = util.split_h5_name(policy_state_str)
    print('Loading policy parameters from %s in %s' % (policy_key, policy_file))
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args'])

    # Initialize the MDP
    print('Loading environment', env_name)
    mdp = rlgymenv.RLGymMDP(env_name)
    print('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    # Initialize the policy
    nn.reset_global_scope()
    # 根据参数确定是否需要将环境变量标准化，是环境变量具有方差为0，均值为1的正态分布特性
    enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none' # 是否将环境变量正则化
    # 如果这个task的动作空间是一个连续变量
    if isinstance(mdp.action_space, policyopt.ContinuousSpace):
        # 把每一动作的概率分布定义为一个高斯分布
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'], # 设置策略的隐藏层
            min_stdev=0.,
            init_logstdev=0.,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            enable_obsnorm=enable_obsnorm)
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')

    # Load the policy parameters
    policy.load_h5(policy_file, policy_key)

    return mdp, policy, train_args


def exec_saved_policy(env_name, policystr, num_trajs, template_obs,template_a,deterministic, max_traj_len=None):
    import policyopt
    from policyopt import SimConfig, rl, util, nn, tqdm
    from environments import rlgymenv
    import gym

    # Load MDP and policy
    mdp, policy, _ = load_trained_policy_and_mdp(env_name, policystr)
    # max_traj_len = min(mdp.env_spec.timestep_limit, max_traj_len) if max_traj_len is not None else mdp.env_spec.timestep_limit
    max_traj_len = min(mdp.env_spec.max_episode_steps, max_traj_len) if max_traj_len is not None else mdp.env_spec.max_episode_steps

    print('Sampling {} trajs (max len {}) from policy {} in {}'.format(num_trajs, max_traj_len, policystr, env_name))

    trajbatchs = np.empty([num_trajs,max_traj_len],dtype=float)
    templates = zip(template_obs,template_a)
    j = 0
    for template,action in templates:
        # print(template.shape)
        d = np.zeros(template.shape[0])
        i = 0
        temp = zip(template,action)
        for obs,a in temp:
            obs = obs.reshape(1,obs.shape[0])
            test_a,test_adist = policy.sample_actions(obs,deterministic)
            test_a = test_a[0]
            distance = lambda x,y: norm(a-test_a)
            d[i] = compute_dtw(a,test_a)
            i+=1
        trajbatchs[j] = d
        j+=1
    return trajbatchs,policy, mdp

def compute_dtw(example_seq,test_seq):
    example = example_seq.reshape(-1,1)
    test = test_seq.reshape(-1,1)
    manhattan_distance = lambda x,y: norm(example-test)
    d,cost_matrix,acc_cost_matrix,path = dtw(example,test,dist=manhattan_distance)
    return d

def test():
    example = np.random.randn(3)
    template = np.random.randn(3)
    d = compute_dtw(example,template)
    print(d)
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('template_traj',type=str)
    parser.add_argument('test_traj',type=str)
    parser.add_argument('test_policy',type=str)
    parser.add_argument('env_name',type=str,default="Hopper-v2")
    args = parser.parse_args()
    example_obs,example_a,example_r,example_len = readtrajs(args.template_traj)
    # print(example_r.sum(axis=1))
    test_obs,test_a,test_r,test_len = readtrajs(args.test_traj)
    distance,policy, mdp = exec_saved_policy(args.env_name,args.test_policy, 1000, example_obs,example_a,False)
    print(distance.sum(axis=1).mean())
    # example = np.concatenate([example_obs,example_a],axis=2)
    # test = np.concatenate([test_obs,test_a],axis=2)
    

if __name__ == "__main__":
    # test()
    main()