# GAIL算法流程

## 初始化定义

1. 定义MDP过程，为了后续进行蒙特卡洛模拟
```python
mdp = rlgymenv.RLGymMDP(args.env_name)
```
2. 定义生成网络(Policy)
```python
policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
```
其中包括：
- 环境正则化过程
```python
normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(obsfeat_B_Df)
```
- 使用网络将obs映射为动作分布参数
```python
actiondist_B_Pa = self._make_actiondist_ops(normalized_obsfeat_B_Df)
```
- 输入动作在新旧策略中的动作分布概率
```python
logprobs_B = self._make_actiondist_logprob_ops(actiondist_B_Pa, input_actions_B_Da)
```
- 利用优势值计算obj和objgrad，作为后续更新网络参数的依据
```python
advantage_B = tensor.vector(name='advantage_B') # 优势函数
impweight_B = tensor.exp(logprobs_B - proposal_logprobs_B)
obj = (impweight_B*advantage_B).mean() # 主观价值
objgrad_P = thutil.flatgrad(obj, param_vars) # 计算obj中参数的梯度
```
- 计算新旧策略动作分布的KL散度，作为判断是否达到停止迭代的标准
```python
kl_B = self._make_actiondist_kl_ops(proposal_actiondist_B_Pa, actiondist_B_Pa) # 计算新旧策略的kl散度
kl = kl_B.mean() # 散度平均值
```
- 定义使用TRPO进行参数更新

3. 载入专家轨迹
4. 定义判别网络(TransitionClassifier)
网络定义包括：
- 将(s,a)对利用网络转换为一个分数，是真实样本的概率
- 定义score与reward的对应关系
```python
rewards_B = -tensor.log(1.-tensor.nnet.sigmoid(scores_B))
```
- 定义损失函数，并利用损失进行参数更新，其中使用adam迭代器进行网络参数更新
```python
loss = ((losses_B - self.ent_reg_weight*ent_B)*weights_B).sum(axis=0)
adamstep_without_time = thutil.function(
                [obsfeat_B_Df, a_B_Da, labels_B, weights_B], loss,
                updates=thutil.adam(loss, param_vars, lr=adam_lr))
```

5. 定义ValueFunc
```python
vf = rl.ValueFunc(
            hidden_spec=args.policy_hidden_spec,
            obsfeat_space=mdp.obs_space,
            enable_obsnorm=args.obsnorm_mode != 'none',
            enable_vnorm=True,
            max_kl=args.vf_max_kl,
            damping=args.vf_cg_damping, # 共轭梯度阻尼
            time_scale=1./mdp.env_spec.timestep_limit,
            varscope_name='ValueFunc')
```
- 使用obs和带有时间序列衰减的t，使用网络计算val，返回的是一个状态值
- 使用新旧策略得到的val作为网络参数更新的依据

6. 定义整体优化器
```python
opt = imitation.ImitationOptimizer(
            mdp=mdp,
            discount=args.discount,
            lam=args.lam,
            policy=policy, # RL
            sim_cfg=policyopt.SimConfig(
                min_num_trajs=-1, min_total_sa=args.min_total_sa,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len),
            step_func=rl.TRPO(max_kl=args.policy_max_kl, damping=args.policy_cg_damping),
            reward_func=reward,
            value_func=vf,
            policy_obsfeat_fn=lambda obs: obs,
            reward_obsfeat_fn=lambda obs: obs,
            policy_ent_reg=args.policy_ent_reg,
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            ex_t=ext_Bstacked)
```

## 迭代更新
1. 轨迹采样，返回[obsfeat,a,actiondist,reward]，这个reward为环境返回的。
2. 计算Advantege value
- 使用判别网络输出score，并将score映射为reward（一个概率值）
- 使用概率值计算advantage
    - 利用概率值计算Q值
    - 使用ValueFunc计算状态值
    - 利用ValueFunc计算得出的状态值，使用GAE方法得到advantage

3. 利用advantage更新生成网络
- 使用由Advantage计算出的obj和objgrad作为更新的依据

- 更新生成网络的环境参数

4. 更新判别网络
- 使用判别网络的损失函数，作为更新的依据

5. 更新ValueFunc
- 使用新的判别网络参数，得到新的reward
- 计算新的Q值
- 通过新的Q值，转换得到新的V值
- 利用新的V值，计算obj，作为网络参数更新的依据，进行参数更新

```python
target_val_B = tensor.vector(name='target_val_B') # 目标状态值
obj = -tensor.square(val_B - target_val_B).mean()
objgrad_P = thutil.flatgrad(obj, param_vars)
# KL divergence (as Gaussian) and its gradient
old_val_B = tensor.vector(name='old_val_B')
kl = tensor.square(old_val_B - val_B).mean()
```
