1. 根据专家策略生成专家轨迹
```bash
python ./scripts/im_pipeline.py ./pipelines/im_pipeline.yaml 0_sampletrajs
```

pipeline:
```yaml
training:
  full_dataset_num_trajs: 50 # 生成的轨迹数量
  dataset_num_trajs: [4, 11, 18, 25] # GAIL载入专家轨迹的数量4条
  deterministic_expert: false # 使用专家策略生成轨迹 是否使用决定性
  runs: 1
```

2. 查看轨迹reward
```bash
python ./scripts/print_saved_returns.py ./imitaion_runs/modern_stohastic/trajs/trajs_hopper.5
```

3. 利用专家轨迹，进行GAIL，生成模拟策略 保存训练数据
```bash
nohup python3 ./scripts/imitate_mj.py --mode ga --env_name Hopper-v2 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=hopper,num_trajs=4,run=0.h5 >imitation_runs/modern_stochastic/logfiles/ga_Hopper-v2_iter_500.log 2>&1
```

4. 查看训练结果
```bash
python3 ./scripts/showlog.py ./imitaion_runs/modern_stochastic/checkpoint_all/alg=ga,task=walker,num_trajs=4,run=0.h5
```

5. 测试
```bash
python3 ./scripts/im_pipeline.py ./pipelines/im_pipeline_v2.yaml 2_eval
```

nohup python3 ./scripts/imitate_mj.py --mode ga --env_name Hopper-v3 --data mix-trajs/trpo_Hopper-v3_step-interval=300-model-num=2-mixture.h5 --limit_trajs 100 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log mix-result/checkpoints_all/alg=ga,task=hopper,num_trajs=100,algo=trpo,model-num=2,interval=300,run=1.h5 > mix-result/logfiles/ga_Hopper-v3_iter_500_limit_trajs=100_algo=trpo_model-num=3-interval=300_1.log 2>&1