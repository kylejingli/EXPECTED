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
import random




matplotlib.rcParams['figure.figsize'] = (5, 5)
matplotlib.rcParams['font.size'] = 14
#matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.serif'] = 'Times'
# matplotlib.rcParams['text.usetex'] = True
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





# Here's a plotting function to visualize our results.
def plot(X, Y, X_, Y_, X1_test, X2_test, py, conf, stage, size=120):    
    ims = []
    cmap = 'Blues'
    
    fig = plt.figure(figsize=(6, 5))
    #x = plt.gca()
    ax = plt.subplot(111)
    ax.set_facecolor('papayawhip')  #whitesmoke

    # Decision boundary contour
    plt.contour(X1_test, X2_test, py.reshape(size, size), levels=[0.5], colors='black', linewidths=[3])
    
    # Background shade, representing confidence
    conf = np.clip(conf, 0, 0.999999)
    #im = plt.contourf(X1_test, X2_test, conf.reshape(size, size), alpha=0.7, 
    #          levels=np.arange(0.5, 1.01, 0.1), cmap=cmap, vmin=0.5, vmax=1)
    #plt.colorbar(im)

    # Scatter plot the training data
    s = 20*6
    t1 = plt.scatter(Xt[Yt==0][:, 0], Xt[Yt==0][:, 1], s=s, c='mistyrose', edgecolors='silver', linewidths=0.5, marker='v')
    t2 = plt.scatter(Xt[Yt==1][:, 0], Xt[Yt==1][:, 1], s=s, c='lightsteelblue', edgecolors='silver', linewidths=0.5, marker='v')
    s1 = plt.scatter(X[Y==0][:, 0], X[Y==0][:, 1], s=s, c='coral', edgecolors='k', linewidths=0.5) # c='mistyrose'  edgecolors='silver'
    s2 = plt.scatter(X[Y==1][:, 0], X[Y==1][:, 1], s=s, c='blue', edgecolors='k', linewidths=0.5) #c='lightsteelblue'
    #plt.plot(X[Y==0][:, 0], X[Y==0][:, 1], c='coral', linewidth=0.5, marker='o', label='source class 1')
    #plt.plot(X[Y==1][:, 0], X[Y==1][:, 1], c='blue', linewidth=0.5, marker='o', label='source class 2')
    #plt.plot(Xt[Yt==0][:, 0], Xt[Yt==0][:, 1], c='coral', linewidth=0.5, marker='v', label='test class 1')
    #plt.plot(Xt[Yt==1][:, 0], Xt[Yt==1][:, 1], c='blue', linewidth=0.5, marker='v', label='test class 2')
    
    plt.xlim(test_range)
    plt.ylim(test_range)
    plt.xticks([])
    plt.yticks([])
    
    plt.legend((s1, s2, t1, t2),
           ('source pos', 'source neg', 'target pos', 'target neg'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           borderpad=0.3,
           fontsize=14)
    plt.savefig('toy_'+stage+'.pdf', dpi=600, bbox_inches='tight')

def sigmoid(x):
    return 1/(1+np.exp(-x))



test_range = (0, 10)
test_rng = np.linspace(*test_range, 50)
X1_test, X2_test = np.meshgrid(test_rng, test_rng)
X_test = np.stack([X1_test.ravel(), X2_test.ravel()]).T
X_test = torch.from_numpy(X_test).float()
model.eval()
with torch.no_grad():
     model.eval()
     py = sigmoid(model(X_test).squeeze().numpy())
    
conf = np.maximum(py, 1-py)
stage = 'source'
plot(Xs, Ys, Xt, Yt, X1_test, X2_test, py, conf, stage, size=50)

# This is the confidence plot (defined as the maximum probability of each prediction) of the model. 
# This is what we are familiar with and use all the time in deep learning. 
# Notice how the confidence is high (close to one) almost everywhere. Do you think this is a good prediction? Why or why not?


for name, p in model.named_parameters():
    print(name)
    print(p)


def accuracy(x, y):
    with torch.no_grad():
         model.eval()
         y_pre = sigmoid(model(x).squeeze())
         #print(model(x).squeeze())
         y_pre[y_pre>0.5] = 1
         y_pre[y_pre<=0.5] = 0
         return (y_pre==y).sum().item()/size


train_acc = accuracy(X_train,y_train)
print(f'source accuracy {train_acc:.3f}')

X_target = torch.from_numpy(Xt).float()
Y_target = torch.from_numpy(Yt).float()
target_acc = accuracy(X_target,Y_target)      
print(f'target accuracy {target_acc:.3f}')
        
# collect model parameters
'''
paras = []
for parameters in model.parameters():
    w = ult.parameters_to_vector(parameters)
    paras.append(w)
    
z = torch.cat((paras[0],paras[1]),dim=0)
print(z)
'''
z = torch.zeros(h)
z = model.clf.weight
#z = torch.reshape(model.clf.weight)


# updating the model by limited query

'''
for it in range(20):
    grad = gradient_estimate(z,xt,yt)
    z = z-lr * grad
    model.clf.weight.data = z[0:1]
    model.clf.bias.data = z[2]
    target_acc = accuracy(X_target,Y_target)
    print(f'iter={it},target accuracy {target_acc:.3f}')
'''    

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
# query and use the returned the acc for updating model paras
sigma = 1
B = 10 
lr = 100
eta = 0.01
for it in range(8):
    noise_pos = torch.randn(B//2, z.shape[1])
    noise = torch.cat((noise_pos, -noise_pos), dim=0)
    eval_points = z + sigma * noise
    returned_acc = torch.zeros(B)
    for b in range(B):
        model.clf.weight.data = torch.reshape(eval_points[b,:],(1,-1))
        returned_acc[b] = accuracy(X_target,Y_target)
    # normalize
    mean = torch.mean(returned_acc)
    std = torch.std(returned_acc)
    returned_acc_nor = (returned_acc-mean)/std
    returned_acc_tiled = torch.reshape(returned_acc,(-1,1)).repeat(1, h) 
    #returned_acc_tiled = torch.reshape(returned_acc_nor,(-1,1)).repeat(1, h) 
    
    grad_estimate = torch.mean(returned_acc_tiled * noise/sigma, 0)
    #update z
    lr*=0.999
    #rand_idx = B-1
    #rand_idx = random.randint(0, B)
    #z = torch.reshape(eval_points[rand_idx,:], (1,-1))
    z = z + lr * grad_estimate #- eta*2*z   # 1:20 for original result
    # test on \mu
    model.clf.weight.data = torch.reshape(z,(1,-1))
    acc_mu = accuracy(X_target,Y_target)
    print(f'iter={it},target accuracy {acc_mu:.3f}')

model.eval()
with torch.no_grad():
     model.eval()
     py = sigmoid(model(X_test).squeeze().numpy())
    
conf = np.maximum(py, 1-py)
stage='target'
#plot(Xs, Ys, Xt, Yt, X1_test, X2_test, py, conf, stage, size=50 )    
