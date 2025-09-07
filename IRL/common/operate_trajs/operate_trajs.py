import sys
sys.path.append("/root/f-IRL/baselines/operate_trajs/")

## 原格式
# import h5py

# f = h5py.File("trpo-ppo-ddpg-sac-Hopper-v3_step-interval=20-model-num=4-mixture.h5", "r")

# for key in f.keys():
#     print("key=",key)
#     print(f[key].name)  #字典中key
#     # print(f[key][:])

    
## IRL格式
import torch
pt_file = torch.load("/root/f-IRL/expert_data/states/HopperFH-v0.pt")
print(pt_file[:5])
