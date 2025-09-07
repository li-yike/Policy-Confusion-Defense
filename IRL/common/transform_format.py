### operate_trajs.py ###
########################

# import sys
# sys.path.append("/root/f-IRL/conmmon/")

# ## 原格式
# import h5py

# f = h5py.File("4-sac-HalfCheetah-lambda=0.3-n_trajs=100-iter=400000.h5", "r")

# for key in f.keys():
#     print("key=",key)
#     print(f[key].name)  #字典中key
#     # print(f[key][:])

    
# ## IRL格式
# import torch
# pt_file = torch.load("/root/autodl-tmp/expert_data_irl/states/HopperFH-v0.pt")
# print(pt_file[:5])



### transform_format.py ###
###########################
# python ./transform_format.py 4-sac-HalfCheetah-lambda=0.3-n_trajs=100-iter=400000.h5 HalfCheetahFH-v0 RCD
# /root/autodl-tmp/expert_data_irl/states/

import h5py
import torch
import sys
import os
sys.path.append("/root/f-IRL/conmmon/")

if __name__ == "__main__":
    print('ok')
    f = h5py.File(sys.argv[1],'r')
    env_name = sys.argv[2]
    obs = f['obs_B_T_Do']
    obs_tensor = torch.tensor(obs[:])
    f.close()
    if os.path.exists('/root/autodl-tmp/expert_data_irl/states')==False:
        os.mkdir('/root/autodl-tmp/expert_data_irl')
        os.mkdir('/root/autodl-tmp/expert_data_irl/states')
    if sys.argv[3]:
        algo = sys.argv[3]
        torch.save(obs_tensor,f'/root/autodl-tmp/expert_data_irl/states/{algo}_{env_name}.pt')
    else:
        torch.save(obs_tensor,f'/root/autodl-tmp/expert_data_irl/states/{env_name}.pt')
    print("ok")