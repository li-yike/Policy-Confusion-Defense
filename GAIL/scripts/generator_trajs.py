import argparse
import json
import h5py
import numpy as np
import sys
sys.path.append("/root/code/imitation/")
#import yaml
import os, os.path, shutil
from policyopt import util
# from torchgen.gen import LineLoader
#from yaml import Loader

def load_trained_policy_and_mdp(env_name, policy_state_str):
    import gym
    import policyopt
    from policyopt import nn, rl
    from environments import rlgymenv

    # Load the saved state
    policy_file, policy_key = util.split_h5_name(policy_state_str)
    print('Loading policy parameters from %s in %s' % (policy_key, policy_file))
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args']) # reading expert policy args

    # Initialize the MDP
    print('Loading environment', env_name)
    mdp = rlgymenv.RLGymMDP(env_name)
    print('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    # Initialize the policy
    nn.reset_global_scope()
    enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none'
    if isinstance(mdp.action_space, policyopt.ContinuousSpace): 
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
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


def gen_taskname2outfile(env_name,savepath,trajs_num,filesuffix, run,assert_not_exists=False):
    '''
    Generate dataset filenames for each task. Phase 0 (sampling) writes to these files,
    phase 1 (training) reads from them.
    '''
    taskname2outfile = {}
    trajdir = savepath
    util.mkdir_p(trajdir)
    fname = os.path.join(trajdir, 'trajs_task={}_trajs_num={}_run={}_{}.h5'.format(env_name,trajs_num,run,filesuffix))
    if assert_not_exists:
        assert not os.path.exists(fname), 'Traj destination {} already exists'.format(fname)
    taskname2outfile = fname
    return taskname2outfile



def exec_saved_policy(env_name, policystr, num_trajs, deterministic, max_traj_len=None):
    import policyopt
    from policyopt import SimConfig, rl, util, nn, tqdm
    from environments import rlgymenv
    import gym

    # Load MDP and policy
    mdp, policy, _ = load_trained_policy_and_mdp(env_name, policystr)
    # max_traj_len = min(mdp.env_spec.timestep_limit, max_traj_len) if max_traj_len is not None else mdp.env_spec.timestep_limit
    max_traj_len = min(mdp.env_spec.max_episode_steps, max_traj_len) if max_traj_len is not None else mdp.env_spec.max_episode_steps

    print('Sampling {} trajs (max len {}) from policy {} in {}'.format(num_trajs, max_traj_len, policystr, env_name))

    # Sample trajs
    trajbatch = mdp.sim_mp(
        policy_fn=lambda obs_B_Do: policy.sample_actions(obs_B_Do, deterministic),
        obsfeat_fn=lambda obs:obs,
        cfg=policyopt.SimConfig(
            min_num_trajs=num_trajs,
            min_total_sa=-1,
            batch_size=None,
            max_traj_len=max_traj_len))

    return trajbatch, policy, mdp


def eval_snapshot(env_name, checkptfile, snapshot_idx, num_trajs, deterministic):
    policystr = '{}/snapshots/iter{:07d}'.format(checkptfile, snapshot_idx)
    trajbatch, _, _ = exec_saved_policy(
        env_name,
        policystr,
        num_trajs,
        deterministic=deterministic,
        max_traj_len=None)
    returns = trajbatch.r.padded(fill=0.).sum(axis=1)
    lengths = np.array([len(traj) for traj in trajbatch])
    util.header('{} gets return {} +/- {}'.format(policystr, returns.mean(), returns.std()))
    return returns, lengths


def phase0_sampletrajs(policy,env_name,savepath,num_trajs,filesuffix,run):
    '''
    spec: [
    {'name': 'hopper', 'env': 'Hopper-v3', 'policy': 'expert_policies/modern/log_Hopper-v0_3.h5/snapshots/iter0000500', 'data_subsamp_freq': 20, 'cuts_off_on_success': False}, 
    {'name': 'walker', 'env': 'Walker2d-v3', 'policy': 'expert_policies/modern/walker_eb5b2e_1.h5/snapshots/iter0000480', 'data_subsamp_freq': 20, 'cuts_off_on_success': False}, 
    {'name': 'ant', 'env': 'Ant-v3', 'policy': 'expert_policies/modern/log_Ant-v1_0.h5/snapshots/iter0000500', 'data_subsamp_freq': 20, 'cuts_off_on_success': False}, 
    {'name': 'halfcheetah', 'env': 'HalfCheetah-v3', 'policy': 'expert_policies/modern/log_HalfCheetah-v0_2.h5/snapshots/iter0000500', 'data_subsamp_freq': 20, 'cuts_off_on_success': False}
    ]
    '''
    util.header('=== Phase 0: Sampling trajs from expert policies ===')

    num_trajs = num_trajs
    util.header('Sampling {} trajectories'.format(num_trajs))

    # Make filenames and check if they're valid first
    taskname2outfile = gen_taskname2outfile(env_name,savepath,num_trajs,filesuffix ,run, assert_not_exists=True)

    # Sample trajs for each task
        # Execute the policy
    trajbatch, policy, _ = exec_saved_policy(
        env_name, policy, 100,
        deterministic=False,
        max_traj_len=None)

    # Quick evaluation
    returns = trajbatch.r.padded(fill=0.).sum(axis=1)
    avgr = trajbatch.r.stacked.mean()
    lengths = np.array([len(traj) for traj in trajbatch])
    ent = policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean()
    print('ret: {} +/- {}'.format(returns.mean(), returns.std()))
    print('avgr: {}'.format(avgr))
    print('len: {} +/- {}'.format(lengths.mean(), lengths.std()))
    print('ent: {}'.format(ent))

        # Save the trajs to a file
    with h5py.File(taskname2outfile, 'w') as f:
        def write(dsetname, a):
            f.create_dataset(dsetname, data=a, compression='gzip', compression_opts=9)
        # Right-padded trajectory data
        write('obs_B_T_Do', trajbatch.obs.padded(fill=0.))
        write('a_B_T_Da', trajbatch.a.padded(fill=0.))
        write('r_B_T', trajbatch.r.padded(fill=0.))
        # Trajectory lengths
        write('len_B', np.array([len(traj) for traj in trajbatch], dtype=np.int32))
        # # Also save args to this script
        # argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
        # f.attrs['args'] = argstr
    util.header('Wrote {}'.format(taskname2outfile))

def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('policy', type=str)
    parser.add_argument('--task',type=str,default='Hopper-v3')
    parser.add_argument('--trajs_num',type=int,default=100)
    parser.add_argument('--savepath',type=str,default="result-trajs")
    parser.add_argument('--filesuffix',type=str,default="mix")
    parser.add_argument('--run',type=int,default=0)
    args = parser.parse_args()
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)

    phase0_sampletrajs(args.policy,args.task,args.savepath,args.trajs_num,args.filesuffix,args.run)

if __name__ == '__main__':
    main()
