# EXPECTED
The code is for the publication of TPAMI-23: Earning extra performance from restrictive feedbacks. Notably, here I also present some understandings which are not elaborately stated in the paper. 

## Problem setup
![alt text](https://github.com/kylejingli/EXPECTED/blob/main/figs/EXPECTED%20Problem.png)
Given the initially provided model $F_{{\theta}_0}$,The objective of EXPECTED is to solve the following problem

$$\theta_*=arg\max E(\mathcal{D};F_{\theta}), s.t. queries \le Q$$

## Optimization
1. The key optimization idea is to estimate model gradients via query-feedback pairs based on [NES](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) which is named by performance-guided parameter search (PPS) in our paper. One can alternatively use [CMA-ES](https://pypi.org/project/cmaes/) which is comparable according to our experiments but more heuristic. 

    Here is an example of how the estimated gradient $\nabla\mathbb{E}[E(\theta)]$ approximates the true gradient $\nabla E(\theta)$. The pink arrow denotes the projection of $\nabla E(\theta)$ onto selected finite bases $\epsilon_1$ and $\epsilon_2$. One can easily verify that a true gradient $(2,1,1)$ under this decomposition corresponds to an estimated gradient of $(2,1.1,0)$.
<div align=center>
<img src="https://github.com/kylejingli/EXPECTED/blob/main/figs/gradient%20estimation.png" width="400">
</div>

2. For efficiency, finding only partial layers of DNNs to tune (because of the tight query budget) is another key point of our method. This can be formulated as a multi-arm bandit (MB) problem and the goal is to achieve a lower regret. Please refer to the layerwise coordinate parameter search (LCPS) algorithm of our paper. This is an online process although one can simply understand we are trying to identify the layer importance as the below figure.
<div align=center>
<img src="https://github.com/kylejingli/EXPECTED/blob/main/figs/layer%20importance.png" width="400">
</div>

    Note: For readers who are not quite familiar with the difference between black-box optimization (BO) and reinforcement learning (RL), we recommend referring to the appendices of our paper. Here we simply show their connections if you have some prior knowledge about them.
<div align=center>
<img src="https://github.com/kylejingli/EXPECTED/blob/main/figs/BO_RL.png" width="400">
</div>
   

## Toy example

## Privacy concern 

## New perspective of black-box 

## Code usage
