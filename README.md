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

3. Note: For readers who are not quite familiar with the difference between black-box optimization (BO) and reinforcement learning (RL), we recommend referring to the appendices of our paper. Here we simply show their connections if you have some prior knowledge about them.
<div align=center>
<img src="https://github.com/kylejingli/EXPECTED/blob/main/figs/BO_RL.png" width="400">
</div>
   

## Toy example
Try the toy example where a pre-trained model can adapt to the unseen target data with 80-time queries.
<div align=center>
<img src="https://github.com/kylejingli/EXPECTED/blob/main/figs/toy%20example.png" width="400">
</div>

## Connection to black-box adversarial attack
We are tuning a white-box model through black-box optimization because "we put data target data into a black box". One can see EXPECTED is symmetric to query-based black-box adversarial attack.
<div align=center>
<img src="https://github.com/kylejingli/EXPECTED/blob/main/figs/Connection%20to%20BB%20adv%20attack.png" width="700">
</div>

## Code usage
1. One can start with the toy example to understand why the proposed algorithm works.
2. Adult dataset is used as a binary classification on which we also tested how the model adapts if some fairness metric is needed on downstream tasks.
3. CIFAR-10-C and STS-B are mainly testing the efficacy of LCPS where only contributive layers are identified in an online optimization manner.

## Citation
If you find this repository helpful to your study, please do not forget to cite [it].
    @article{li2023earning,
  title={Earning Extra Performance from Restrictive Feedbacks},
  author={Li, Jing and Pan, Yuangang and Lyu, Yueming and Yao, Yinghua and Sui, Yulei and Tsang, Ivor W},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
