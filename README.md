# Policy-Confusion-Defense
This is the code for "Towards Preventing Imitation Learning Attack via Policy Confusion Defense"

## Abstract
In real-world reinforcement learning (RL) applications, privacy protection for commercially valuable policies has been highlighted due to their vulnerability to imitation learning (IL) attacks, where attackers can approximate near-optimal policies only from their observations.
Existing defense methods, including generating adversarial trajectories and training privacy-preserving RL policies, suffer from limitations such as scenario specificity and limited effectiveness.
To tackle the limitations, we model the privacy-preserving policy training as the problem of finding an optimal dynamical switching among a diverse set of expert policies, aiming to induce the confusion for potential imitation attackers. Thus, we propose a hierarchical Policy Confusion Defense (PCD) framework that incorporates multiple near-optimal expert policies at the low level and employs a switcher policy at the high level. 
Furthermore, to accommodate various scenarios, two high-level switcher policies are developed: a random-based switcher with low resource requirements, and an adversarial-based switcher optimizing the low-level policy selection with dual objectives on task-capability and defense-effectiveness.
Extensive experiments are conducted to evaluate our defense framework against three imitation attacks, compared to two state-of-the-art defense methods. The results demonstrate that our framework is effective in (a) degrading the approximated policies by imitation attackers while (b) maintaining good performance in the original RL tasks.
