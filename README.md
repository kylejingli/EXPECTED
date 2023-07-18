# EXPECTED
The code is for the publication of TPAMI-23: Earning extra performance from restrictive feedbacks. Notably, here I also present some understandings which are not elaborately stated in the paper. 

## Problem setup
![alt text](https://github.com/kylejingli/EXPECTED/blob/main/figs/EXPECTED%20Problem.png)
Given the initially provided model $F_{{\theta}_0}$,The objective of EXPECTED is to solve the following problem

$$\theta_*=arg\max E(\mathcal{D};F_{\theta}), s.t. queries \le Q$$

## Optimization
1. The key optimization idea is to estimate model gradients via query-feedback pairs based on [NES](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf). One can alternatively use [CMA-ES](https://pypi.org/project/cmaes/) which is comparable according to our experiments but more heuristic. 

Here is an example of how the estimated gradient $\nabla\mathbb{E}[E(\theta)]$ approximates the true gradient $\nabla E(\theta)$. The pink arrow denotes the projection of $\nabla E(\theta)$ onto selected finite bases $\epsilon_1$ and $\epsilon_2$. One can easily verify that a true gradient $(2,1,1)$ under this decomposition corresponds to an estimated gradient of $(2,1.1,0)$.
<div align=center>
<img src="https://github.com/kylejingli/EXPECTED/blob/main/figs/gradient%20estimation.png" width="300">
</div>

2. For efficiency, finding only partial layers of DNNs to optimize (because of the tight query budget) is another key point of our method. This can be formulated as a multi-arm bandit problem.
