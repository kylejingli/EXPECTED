# toy examples for verification of query-based model adaptation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as ult
import numpy as np
import matplotlib
import matplotlib.cm as cm
from sklearn import datasets
from math import *
import seaborn as sns; sns.set_style('white')


matplotlib.rcParams['figure.figsize'] = (5, 5)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.serif'] = 'Times'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['lines.linewidth'] = 1.0
plt = matplotlib.pyplot


# First, we create a toy binary classification dataset, including source and target.
np.random.seed(7777)
size = 90
train_range_s = (2, 6)
train_range_t = (4, 6)
Xs, Ys = datasets.make_blobs(n_samples=size, centers=2, cluster_std=0.7, 
               center_box=train_range_s, random_state=62)
Xt, Yt = datasets.make_blobs(n_samples=size, centers=2, cluster_std=[[0.1, 1.5], [0.1, 1.5]], 
               center_box=train_range_t, random_state=62)


# We will train a linear classifier on source data

torch.manual_seed(7777)

m, n = Xs.shape
h = 20

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.ReLU(), 
            nn.Linear(h, h), 
            nn.BatchNorm1d(h),
            nn.ReLU()
        )
        self.clf = nn.Linear(h, 1, bias=False)
    
    def forward(self, x):
        x = self.feature_map(x)
        return self.clf(x)
    
    
X_train = torch.from_numpy(Xs).float()
y_train = torch.from_numpy(Ys).float()
    
model = Model()
opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

for it in range(5000):
    y_pred = model(X_train).squeeze()
    l = F.binary_cross_entropy_with_logits(y_pred, y_train)
    l.backward()
    opt.step()
    opt.zero_grad()
    
print(f'Loss: {l.item():.3f}')

torch.save(model.state_dict(),'toy_model.pt')



 
      
    

    

'''
# Now, let's see whether we can incorporate the model's uncertainty into the prediction. That is, we want to infer the posterior, and use it to marginalize the prediction:
# 
# $$
#     p(y = 1 \mid x) = \int \sigma(f_\theta(x)) \, p(\theta \mid D) \, d\theta \, , 
# $$
# 
# where $f_\theta$ is our ReLU network.
# 
# One simple way to do this is by doing a Laplace approximation: We approximate the posterior as $\mathcal{N}(\theta \mid \theta_\text{MAP}, H^{-1})$ where $\theta_\text{MAP}$ is the trained weights and $H := -\nabla^2_\theta \log p(\theta \mid D)$ is the Hessian of the negative log-posterior.
# 
# However, notice that if the network is large (has many parameters), then $H$ is huge since it scales quadratically to the number of parameters. Here, we make another simplifying assumption: We only apply the Laplace approximation at the _last-layer_. Assume that $\phi$ is the first $L-1$ layers of the network and $w^L$ is the last-layer's weight vector. Then, the prediction is given by:
# 
# $$
#     p(y = 1 \mid x) \approx \int \sigma({w^L}^\top \phi(x)) \, \mathcal{N}(w^L \mid w^L_\text{MAP}, \Sigma) \, dw^L \, ,
# $$
# 
# where $\Sigma = (-\nabla^2_{w^L} \log p(\theta \mid D))^{-1}$ is the inverse of the particular sub-matrix of $H$ corresponding to the Hessian of the last-layer.
# 
# The previous integral is intractable, so we need a further approximation. One can use the so-called "probit approximation" [1]. First, let's notice that we have an induced Gaussian over the "pre-sigmoid" network's output is $\mathcal{N}({w^L}^\top \phi(x) \mid m, v)$ where $m(x) := {w^L_\text{MAP}}^\top \phi(x)$ is the usual MAP prediction and $v(x) := \phi(x)^\top \Sigma \phi(x)$ is the variance. Then, the probit approximation is simply:
# 
# $$
#     p(y = 1 \mid x) \approx \sigma \left( \frac{m(x)}{\sqrt{1 + \pi/8 \, v(x)}} \right) \, .
# $$
# 
# 
# [1] MacKay, David JC. "The evidence framework applied to classification networks." Neural computation 4.5 (1992): 720-736.

# In[10]:


# Exact Hessian using PyTorch's autograd. Credits due to Felix Dangel.
from hessian import exact_hessian


W = list(model.parameters())[-1]
shape_W = W.shape

w_map = W.view(-1).data.numpy()


def neg_log_posterior(var0):
    # Negative-log-likelihood
    nll = F.binary_cross_entropy_with_logits(model(X_train).squeeze(), y_train, reduction='sum')
    # Negative-log-prior
    nlp = 1/2 * W.flatten() @ (1/var0 * torch.eye(W.numel())) @ W.flatten()
    
    return nll + nlp


def get_covariance(var0):
    # Outputs the inverse-Hessian of the negative-log-posterior at the MAP estimate
    # This is the posterior covariance
    loss = neg_log_posterior(var0)
    Lambda = exact_hessian(loss, [W])  # The Hessian of the negative log-posterior
    Sigma = torch.inverse(Lambda).detach().numpy()
    
    return Sigma


@torch.no_grad()
def predict(x, Sigma):    
    phi = model.feature_map(x).numpy()  # Feature vector of x
    m = phi @ w_map  # MAP prediction

    # "Moderate" the MAP prediction using the variance (see MacKay 1992 "Evidence Framework ...")
    # This is an approximation of the expected sigmoid (the so-called "probit approximation")
    v = np.diag(phi @ Sigma @ phi.T)
    py = sigmoid(m/np.sqrt(1 + pi/8 * v))
    
    return py


# The weight decay used for training is the Gaussian prior's precision
# Here we work in term of variance, which is just 1/precision.
var0 = 1/5e-4 

# Get the posterior covariance and make prediction
Sigma = get_covariance(var0)
py = predict(X_test, Sigma)

conf = np.maximum(py, 1-py)

plot(X, Y, X1_test, X2_test, py, conf, size=size)


# In[ ]:
'''



