1. Gnerate Expert Demonstration 
```bash
python ./scripts/im_pipeline.py ./pipelines/im_pipeline.yaml 0_sampletrajs
```

pipeline:
```yaml
training:
  full_dataset_num_trajs: 50 # Number of generated trajectories
  dataset_num_trajs: [4, 11, 18, 25] # Load Trajectories
  deterministic_expert: false 
  runs: 1
```

2. Check Trajectory Rewards
```bash
python ./scripts/print_saved_returns.py ./imitaion_runs/modern_stohastic/trajs/trajs_hopper.5
```

3. GAIL training on Expert Demonstration
```bash
nohup python3 ./scripts/imitate_mj.py --mode ga --env_name Hopper-v2 --data imitation_runs/modern_stochastic/trajs/trajs_hopper.h5 --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=hopper,num_trajs=4,run=0.h5 >imitation_runs/modern_stochastic/logfiles/ga_Hopper-v2_iter_500.log 2>&1
```

4. Check Training Results
```bash
python3 ./scripts/showlog.py ./imitaion_runs/modern_stochastic/checkpoint_all/alg=ga,task=walker,num_trajs=4,run=0.h5
```

5. Testing
```bash
python3 ./scripts/im_pipeline.py ./pipelines/im_pipeline_v2.yaml 2_eval
```

nohup python3 imitate_mj.py --mode ga --env_name Hopper-v3 --data  --limit_trajs 4 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --min_total_sa 50000 --max_iter 501 --reward_include_time 0 --reward_lr .01 --log imitation_runs/modern_stochastic/checkpoints_all/alg=ga,task=hopper,num_trajs=4,run=0.h5 >imitation_runs/modern_stochastic/logfiles/ga_Hopper-v2_iter_500.log 2>&1