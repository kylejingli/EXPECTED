# Adult experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as ult
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import math
from cmaes import CMA, CMAwM, SepCMA
import time

mydir = '/sparse_version/data'
h = 80
Q=1000
B = int(h/4)
IT = int(Q/B)
# We rebuild Adult dataset, making US as source domain while others are target domain
# Run Adult_preprocess_sparse.py. To verify the model generalization, we further split the NonUS into two parts
class Model(nn.Module):
    def __init__(self,n,h):
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

def train():    
    Xtr = np.load(mydir+'Xtr.npy')
    Ytr = np.load(mydir+'Ytr.npy')
    n = Xtr.shape[1]
    X_train = torch.from_numpy(Xtr).float()
    Y_train = torch.from_numpy(Ytr).float()
        
    model = Model(n,h)
    opt = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    
    EPOCH = 400
    batch_size = 1024
    batch_num = X_train.shape[0]//batch_size
    for epoch in range(EPOCH):
        for batch_idx in range(batch_num+1):
            if batch_idx == batch_num:
               x_b = X_train[batch_idx*batch_size:X_train.shape[0],:]
               y_b = Y_train[batch_idx*batch_size:X_train.shape[0]]
            else:
                x_b = X_train[batch_idx*batch_size:(batch_idx+1)*batch_size,:]
                y_b = Y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y_b_pred = model(x_b).squeeze()
            l = F.binary_cross_entropy_with_logits(y_b_pred, y_b)
            l.backward()
            opt.step()
            opt.zero_grad()
        print(f'Loss: {l.item():.3f}')
        
    model_path = '/adult_mlp'
    torch.save(model,model_path)
    train_acc = accuracy(model,X_train,Y_train)
    print(f'training accuracy {train_acc:.3f}') 
    
# split Xte into two parts for verification and testing respectively
def split(x, y):
    random.seed()
    #shuffle
    xy = np.concatenate((x,y[:,np.newaxis]),axis=1)
    np.random.shuffle(xy)
    #split
    n = xy.shape[0]
    k = n//2
    x1 = xy[:k,:-1]
    y1 = xy[:k,-1]
    x2 = xy[k:n,:-1]
    y2 = xy[k:n,-1]
    return x1, y1, x2, y2

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def accuracy(model, x, y):
    with torch.no_grad():
         model.eval()
         y_pre = sigmoid(model(x).squeeze())
         y_pre[y_pre>0.5] = 1
         y_pre[y_pre<=0.5] = 0
         return round((y_pre==y).sum().item()/x.shape[0],4)

def standard_finetuning(model, X_test1, Y_test1, X_test2, Y_test2):
    model_q = copy.deepcopy(model)
    test_acc1 = accuracy(model_q,X_test1,Y_test1)
    test_acc2 = accuracy(model_q,X_test2,Y_test2)
    ACC1 = []
    ACC2 = []
    ACC1.append(test_acc1)
    ACC2.append(test_acc2)
    print(f'accuracy1 {test_acc1:.3f}')   
    print(f'accuracy2 {test_acc2:.3f}')

    opt = optim.SGD([model_q.clf.weight], lr=1e-2, momentum=0.9, weight_decay=5e-4)
    EPOCH = 100
    batch_size = 64
    batch_num = X_test1.shape[0]//batch_size
    for epoch in range(EPOCH):
        for batch_idx in range(batch_num+1):
            if batch_idx == batch_num:
               x_b = X_test1[batch_idx*batch_size:X_test1.shape[0],:]
               y_b = Y_test1[batch_idx*batch_size:X_test1.shape[0]]
            else:
                x_b = X_test1[batch_idx*batch_size:(batch_idx+1)*batch_size,:]
                y_b = Y_test1[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y_b_pred = model_q(x_b).squeeze()
            l = F.binary_cross_entropy_with_logits(y_b_pred, y_b)
            l.backward()
            opt.step()
            opt.zero_grad()
        print(f'Loss: {l.item():.3f}')
        ACC1.append(accuracy(model_q,X_test1,Y_test1))
        #print(f'tuning accuracy {tuning_acc:.3f}')
        ACC2.append(accuracy(model_q,X_test2,Y_test2))
        #print(f'test accuracy {test_acc:.3f}') 
    del model_q
    return ACC1, ACC2

def private_finetuning(model, X_test1, Y_test1, X_test2, Y_test2):
    model_q = copy.deepcopy(model)
    test_acc1 = accuracy(model_q,X_test1,Y_test1)
    test_acc2 = accuracy(model_q,X_test2,Y_test2)
    ACC1 = []
    ACC2 = []
    ACC1.append(test_acc1)
    ACC2.append(test_acc2)
    print(f'accuracy1 {test_acc1:.3f}')   
    print(f'accuracy2 {test_acc2:.3f}')
   
    # collect model parameters
    z = torch.zeros(h)
    z = model_q.clf.weight

    # query and use the returned the acc for updating model paras
    sigma = 1 
    lr = 1
    random.seed()
    t = time.time()
    for it in range(IT):
        noise_pos = torch.randn(B//2, z.shape[1])
        noise = torch.cat((noise_pos, -noise_pos), dim=0)
        eval_points = z + sigma * noise
        returned_acc = torch.zeros(B)
        hol_returned_acc = torch.zeros(B)
        for b in range(B):
            model_q.clf.weight.data = torch.reshape(eval_points[b,:],(1,-1))
            returned_acc[b] = accuracy(model_q,X_test1,Y_test1)
            hol_returned_acc[b] = accuracy(model_q,X_test2,Y_test2)
        # normalize
        batch_max, idx = torch.max(returned_acc,dim=0)
        mean = torch.mean(returned_acc)
        std = torch.std(returned_acc)
        returned_acc_nor = torch.nan_to_num( (returned_acc-mean)/std, 1)
        returned_acc_tiled = torch.reshape(returned_acc_nor,(-1,1)).repeat(1, h) 
        #print(f'iter={it},target accuracy {torch.mean(returned_acc):.3f}')
        grad_estimate = torch.mean(returned_acc_tiled * noise/sigma, 0)
        #update z
        #lr decay
        lr*=0.99
        z = z + lr * grad_estimate
        # test on \mu
        model_q.clf.weight.data = torch.reshape(z,(1,-1))
        acc_mu1 = accuracy(model_q,X_test1,Y_test1)
        acc_mu2 = accuracy(model_q,X_test2,Y_test2)
        # three-party comparison
        rr = torch.stack( (torch.tensor(ACC1[-1]), batch_max, torch.tensor(acc_mu1)), 0)
        acc_max, which_one = torch.max(rr,dim=0)
        ACC1.append(acc_max)
        if which_one==0:           
            ACC2.append(ACC2[-1])
        elif which_one==1:
            ACC2.append(hol_returned_acc[idx])
        else:
            ACC2.append(acc_mu2)
    time_cost = time.time()-t
    print(time_cost)
    del model_q   
    return ACC1, ACC2

def static_random_sampling(model, X_test1, Y_test1, X_test2, Y_test2):
    model_q = copy.deepcopy(model)
    test_acc1 = accuracy(model_q,X_test1,Y_test1)
    test_acc2 = accuracy(model_q,X_test2,Y_test2)
    ACC1 = []
    ACC2 = []
    ACC1.append(test_acc1)
    ACC2.append(test_acc2)
    print(f'accuracy1 {test_acc1:.3f}')   
    print(f'accuracy2 {test_acc2:.3f}')

    z = torch.zeros(h)
    z = model_q.clf.weight
    NUM = Q
    sigma = 1
    noise = torch.randn(NUM, z.shape[1])
    eval_points = z + sigma * noise
    for i in range(NUM):
        model_q.clf.weight.data = torch.reshape(eval_points[i,:],(1,-1))
        tune_acc = accuracy(model_q,X_test1,Y_test1)
        if tune_acc > ACC1[-1]:
            ACC1.append(tune_acc)
            test_acc = accuracy(model_q,X_test2,Y_test2)
            ACC2.append(test_acc)
        else:
            ACC1.append(ACC1[-1])
            ACC2.append(ACC2[-1])
    del model_q 
    return ACC1, ACC2

def dynamic_random_sampling(model, X_test1, Y_test1, X_test2, Y_test2):   
    model_q = copy.deepcopy(model)
    test_acc1 = accuracy(model_q,X_test1,Y_test1)
    test_acc2 = accuracy(model_q,X_test2,Y_test2)
    ACC1 = []
    ACC2 = []
    ACC1.append(test_acc1)
    ACC2.append(test_acc2)
    ACC1_history_best = test_acc1

    z = torch.zeros(h)
    z = model_q.clf.weight
    sigma = 1
    tune_best = []
    test_best = []
    for it in range(IT):
        noise_pos = torch.randn(B//2, z.shape[1])
        noise = torch.cat((noise_pos, -noise_pos), dim=0)
        eval_points = z + sigma * noise
        returned_acc = torch.zeros(B)
        for b in range(B):
            model_q.clf.weight.data = torch.reshape(eval_points[b,:],(1,-1))
            ACC1.append(accuracy(model_q,X_test1,Y_test1))
            ACC2.append(accuracy(model_q,X_test2,Y_test2)) 
            if accuracy(model_q,X_test1,Y_test1) > ACC1_history_best:
               z = model_q.clf.weight
        tu_current_best = np.amax(ACC1)
        tune_best.append(tu_current_best)
        idx = np.where(ACC1 == tu_current_best)[0]
        test_corr_best = ACC2[idx[0]]
        test_best.append(test_corr_best)
    del model_q   
    return tune_best, test_best
        
def draw(tune, test, sr_tune, sr_test):
    fig = plt.figure(figsize=(6.5,6))
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    fontsize=25
    plt.grid()
    # plot no tuning
    location_x = 0
    location_y_tuningset = tune[0]
    location_y_testset = test[0]
    h1 = plt.scatter(location_x,location_y_tuningset, s=300,marker='v',linewidth=5.0, color=color1, zorder=2, label = 'w/o_tu')
    h2 = plt.scatter(location_x,location_y_testset, s=300,marker='^',linewidth=5.0, color=color2, zorder=2,label = 'w/o_te')
    # plot random searching
    location_x = IT + 1
    location_y_tuningset = sr_tune
    location_y_testset = sr_test
    h3 = plt.scatter(location_x,location_y_tuningset, s=300,marker='1',linewidth=5.0, color=color1, zorder=2,label = 'rand_tu')
    h4 = plt.scatter(location_x,location_y_testset, s=300,marker='2',linewidth=5.0, color=color2, zorder=2, label = 'rand_te')
    # ours
    x = np.arange(IT+1)
    h5 = plt.plot(x,tune,linewidth=5.0,zorder=1,color=color1,linestyle='-',label = 'nes_tu')
    h6 = plt.plot(x,test,linewidth=5.0,zorder=1,color=color2,linestyle='--',label = 'nes_te')

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,3,4,5,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    #plt.legend(['W/O query Tuning','W/O query Test','SRS Tuning','SRS Test','NES Tuning','NES Test'],fontsize=fontsize)
    plt.xlabel('Iteration',fontsize=fontsize)
    plt.ylabel('Accuracy(%)',fontsize=fontsize)
    fig.tight_layout()
    plt.savefig('adult.pdf',dpi=600,format='pdf')

def cma_tuning(model, X_test1, Y_test1, X_test2, Y_test2):
    TestPerformance = []
    TestPerformance.append(accuracy(model,X_test1,Y_test1))
    # collect tuned model parameters
    # tuned_param = torch.zeros(h)
    tuned_params = model.clf.weight.data.clone().detach().squeeze()
    optimizer = CMA(mean=tuned_params.numpy(),sigma=1)
    start_time = time.time()
    for generation in range(math.floor(Q/optimizer.population_size)):
        solutions = []
        acc = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            z = torch.from_numpy(x).to(torch.float32)
            model_q = copy.deepcopy(model)
            model_q.clf.weight.data = torch.reshape(z,(1,-1))
            value = 1.0-accuracy(model_q,X_test1,Y_test1)
            acc.append(1.0-value)
            # test_result = accuracy(model_q,X_test2,Y_test2)
            solutions.append((x,value))
            del model_q
        optimizer.tell(solutions) 
        if max(acc) > TestPerformance[-1]:
           TestPerformance.append(max(acc))
        else:
           TestPerformance.append(TestPerformance[-1]) 
    
    time_cost = time.time()-start_time
    return TestPerformance, time_cost

def cma_wm_tuning(model, X_test1, Y_test1, X_test2, Y_test2):
    TestPerformance = []
    TestPerformance.append(accuracy(model,X_test1,Y_test1))
    tuned_params = model.clf.weight.data.clone().detach().squeeze()
    binary_dim, continuous_dim = 0, h
    bounds = np.concatenate(
                [
                    np.tile([0, 1], (binary_dim, 1)),
                    np.tile([-np.inf, np.inf], (continuous_dim, 1)),
                ]
            )
    steps = np.concatenate([np.ones(binary_dim), np.zeros(continuous_dim)])       
    optimizer = CMAwM(mean=tuned_params.numpy(),sigma=2.0,bounds=bounds, steps=steps)
    start_time = time.time()
    for generation in range(math.floor(Q/optimizer.population_size)):
        solutions = []
        acc = []
        for _ in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()
            z = torch.from_numpy(x_for_eval).to(torch.float32)
            model_q = copy.deepcopy(model)
            model_q.clf.weight.data = torch.reshape(z,(1,-1))
            value = 1.0-accuracy(model_q,X_test1,Y_test1)
            acc.append(1.0-value)
            # test_result = accuracy(model_q,X_test2,Y_test2)
            solutions.append((x_for_tell,value))
            del model_q
        optimizer.tell(solutions) 
        if max(acc) > TestPerformance[-1]:
           TestPerformance.append(max(acc))
        else:
           TestPerformance.append(TestPerformance[-1]) 
        
        if optimizer.should_stop():
            break
    
    time_cost = time.time()-start_time
    return TestPerformance, time_cost

def sep_cma_tuning(model, X_test1, Y_test1, X_test2, Y_test2):
    TestPerformance = []
    TestPerformance.append(accuracy(model,X_test1,Y_test1))
    # collect tuned model parameters
    # tuned_param = torch.zeros(h)
    tuned_params = model.clf.weight.data.clone().detach().squeeze()
    optimizer = SepCMA(mean=tuned_params.numpy(),sigma=1)
    start_time = time.time()
    for generation in range(math.floor(Q/optimizer.population_size)):
        solutions = []
        acc = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            z = torch.from_numpy(x).to(torch.float32)
            model_q = copy.deepcopy(model)
            model_q.clf.weight.data = torch.reshape(z,(1,-1))
            value = 1.0-accuracy(model_q,X_test1,Y_test1)
            acc.append(1.0-value)
            # test_result = accuracy(model_q,X_test2,Y_test2)
            solutions.append((x,value))
            del model_q
        optimizer.tell(solutions) ## this step updates model!!!
        if max(acc) > TestPerformance[-1]:
           TestPerformance.append(max(acc))
        else:
           TestPerformance.append(TestPerformance[-1]) 
        
        if optimizer.should_stop():
            break

    time_cost = time.time()-start_time
    return TestPerformance, time_cost


def main():
    model_path = '/Adult/adult_mlp'
    # classifier training and save model
    training_tag = False
    if training_tag:
        train()    
    model = torch.load(model_path)
    model.eval()
    # load private data
    Xte = np.load(mydir+'Xte.npy')
    Yte = np.load(mydir+'Yte.npy')
    # split
    Xte1, Yte1, Xte2, Yte2 = split(Xte,Yte)
    X_test1 = torch.from_numpy(Xte1).float()
    Y_test1 = torch.from_numpy(Yte1).float()
    X_test2 = torch.from_numpy(Xte2).float()
    Y_test2 = torch.from_numpy(Yte2).float()
    
    result_path = 'Adult/result/'
    
    
    # static random search 
    '''
    for i in range(10):
        sr_tune_best, sr_test_best = static_random_sampling(model,X_test1,Y_test1,X_test2,Y_test2)
        np.save(result_path+'sr_tune_best'+str(i)+'.npy',sr_tune_best)
        np.save(result_path+'sr_test_best'+str(i)+'.npy',sr_test_best)
    # dynamic random search
    for i in range(5):
        dr_tune_best, dr_test_best = dynamic_random_sampling(model,X_test1,Y_test1,X_test2,Y_test2)
        np.save(result_path+'dr_tune_best'+str(i)+'.npy',dr_tune_best)
        np.save(result_path+'dr_test_best'+str(i)+'.npy',dr_test_best)
    '''
    # private tuning
    for i in range(10):
        tuning_acc, test_acc = private_finetuning(model,X_test1,Y_test1,X_test2,Y_test2)
        np.save(result_path+'tuning_acc'+str(i)+'.npy',tuning_acc)
        np.save(result_path+'test_acc'+str(i)+'.npy',test_acc)
    #draw(tuning_acc,test_acc,sr_tune_best,sr_test_corr_best)
    '''
   
    # standard tuning
    tuning_acc, test_acc = standard_finetuning(model,X_test1,Y_test1,X_test2,Y_test2)
    np.save(result_path+'tuning_acc_sft'+'.npy',tuning_acc)
    np.save(result_path+'test_acc_sft'+'.npy',test_acc)
    
    # cma tuning
    for i in range(10):
        tuning_acc, time = cma_tuning(model,X_test1,Y_test1,X_test2,Y_test2)
        np.save(result_path+'cma_tuning_acc'+str(i)+'.npy',tuning_acc)
        np.save(result_path+'cma_time'+str(i)+'.npy',time)
    
 
    for i in range(10):
        tuning_acc, time = cma_wm_tuning(model,X_test1,Y_test1,X_test2,Y_test2)
        np.save(result_path+'cma_wm_tuning_acc'+str(i)+'.npy',tuning_acc)
        np.save(result_path+'cma_wm_time'+str(i)+'.npy',time)
    
    for i in range(10):
        tuning_acc, time = sep_cma_tuning(model,X_test1,Y_test1,X_test2,Y_test2)
        np.save(result_path+'sep_cma_tuning_acc'+str(i)+'.npy',tuning_acc)
        np.save(result_path+'sep_cma_time'+str(i)+'.npy',time)
    '''
if __name__== "__main__":
   main()

   



