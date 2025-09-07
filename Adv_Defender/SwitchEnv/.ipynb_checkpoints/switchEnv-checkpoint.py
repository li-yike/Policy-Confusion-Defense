import copy

import gym
import numpy as np
from gym import spaces


class SwitchEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    # Params：
    # env_name：专家策略运行的原gym环境->str
    # policy：可选择策略的列表 -> list
    def __init__(self, env_name, policy, deterministic, max_steps, MC_traj):
        super().__init__()
        self.env_name = env_name
        self.policy = policy  # 策略列表供step的时候直接执行
        n_action = len(policy)  # 策略数量，也是swtich env的动作空间
        self.env = gym.make(env_name)  # 执行的游戏环境
        self.action_space = spaces.Discrete(n_action)  # 强化学习策略的数量
        self.observation_space = self.env.observation_space  # SwitchEnv的环境与执行的游戏环境相同
        self.observation = self.env.reset()
        self.deterministic = deterministic
        self.max_steps = max_steps
        self.actions = np.zeros([self.max_steps, self.env.action_space.shape[0]])
        self.count = 0
        self.lambd = 0.8 ########
        self.MC_traj = MC_traj
        
    def set_actionProbs(self,probs):
        self.action_probs = probs

    def step(self, action):
        
        # print("Env action Probs: ",self.action_probs)
        
        saved_state = self.env.sim.get_state()  # 保存当前环境的状态
        
        R_clone = 0
        if self.MC_traj > 0:
            for i, attack_policy in enumerate(self.policy):

                prob = self.action_probs[i]  # switch policy选择第i个策略的概率
                attack_reward_list = []

                for j in range(self.MC_traj):
                    attack_env = gym.make(self.env_name)  
                    attack_env.reset()
                    attack_env.sim.set_state(saved_state)  # 新建gym环境，加载当前环境的状态

                    attack_action = attack_policy.predict(self.observation, self.deterministic)
                    _, attack_reward, _, _ = attack_env.step(attack_action[0])


                    attack_reward_list.append(attack_reward)
                # print(f"Policy {i} Prob: {prob}, reward: {np.mean(attack_reward_list)}")
                R_clone += prob * np.mean(attack_reward_list)
            # print("R_clone: ",R_clone)
        
        
        choose_agent = self.policy[action]
        
        policy_action = choose_agent.predict(self.observation, self.deterministic)
        
        self.actions[self.count] = policy_action[0]
        
        self.count += 1

        observation, reward, done, info = self.env.step(policy_action[0])
        
        self.observation = observation

        reward = reward - self.lambd * R_clone

        return observation, reward, done, info

    def reset(self):
        # ...# observation = env.reset()
        observation = self.env.reset()
        self.observation = observation
        self.actions = np.zeros([self.max_steps, self.env.action_space.shape[0]])
        self.count = 0
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        pass

    def close(self):
        pass
