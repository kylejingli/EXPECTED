# EXPECTED
The code is for the publication of TPAMI-23: Earning extra performance from restrictive feedbacks.

## Problem setup
![alt text](https://github.com/kylejingli/EXPECTED/blob/main/figs/EXPECTED%20Problem.png)
Given the initially provided model $F_{{\theta}_0}$,The objective of EXPECTED is to solve the following problem

$$\theta_*=arg\max E(\mathcal{D};F_{\theta}), s.t. queries \le Q$$

## Optimization
The key approach is based on [NES](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) and one can alternatively use [CMA-ES](https://pypi.org/project/cmaes/).
