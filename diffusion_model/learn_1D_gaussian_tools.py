import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from collections import OrderedDict
from torch.func import jacrev, vmap
from matplotlib import cm

# CUDA support 
#device = torch.device('mps')
device = torch.device('cpu')
#device = torch.device('cuda')

class Plain(nn.Module):
    def __init__(self, in_dim , h_dim , out_dim):
        super().__init__()
        self.ln1 = nn.Linear( in_dim , h_dim )
        self.act =nn.Sigmoid()
        self.ln2 = nn.Linear( h_dim , out_dim , bias=True)

    def forward(self, x):
        out = self.ln1(x)
        out = self.act(out)
        out = self.ln2(out)
        return out

def count_parameters(model, requires_grad = True):
    """Count trainable parameters for a nn.Module."""
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def loss_ESM(model, X, s_X):
    return torch.mean((model(X) - s_X)**2)

def generateTrainingPoints_gauss_ESM(N, mu, sigma):
    X = sigma*np.random.randn(N,1) + mu
    s_X = -(X-mu)/(sigma**2)
    
    X_torch = torch.from_numpy(X).requires_grad_(True).float().to(device)
    s_X_torch = torch.from_numpy(s_X).float().to(device)
    return X_torch, s_X_torch

def train_model_gauss_ESM(N, mu, sigma):
    
    N_neurons = 10
    model = Plain(1, N_neurons, 1).to(device)
    print('model = ', model)
    print(f"Number of paramters: {count_parameters(model)}")
    print()
    
    Adam_iter = 10000 # stop when iter > Adam_iter
    update_lr_1 = 5000  # change adam lr from 0.01 to 0.001 at update_lr_1 iteration
    update_lr_2 = 8000  # change adam lr from 0.001 to 0.0001 at update_lr_2 iteration

    optimizerAdam = torch.optim.Adam(
        model.parameters(), 
        lr=0.01
    )

    itera = 0

    savedloss = []
    savedloss_idx = []
    savedloss_valid = []
    savedloss_valid_idx = []
    
    X_train_torch, s_X_train_torch = generateTrainingPoints_gauss_ESM(N, mu, sigma)
    
    step_valid = Adam_iter/10
    step_resample = Adam_iter/100
    
    print('====================================================')
    print('Strat training')
    print()
    print('Adam optimizer, re-sample every ', step_resample, ' steps')
    print()
    
    model.train()
    for step in range(Adam_iter+1):
        if step == update_lr_1:
            print('change learning rate to 0.001')
            for g in optimizerAdam.param_groups:
                g['lr'] = 0.001
        if step == update_lr_2:
            print('change learning rate to 0.0001')
            for g in optimizerAdam.param_groups:
                g['lr'] = 0.0001

        # Backward and optimize
        optimizerAdam.zero_grad()

        lossAdam = loss_ESM(model, X_train_torch, s_X_train_torch)

        savedloss.append(lossAdam.item())
        savedloss_idx.append(itera)

        if step % step_resample == 0:
            X_train_torch, s_X_train_torch = generateTrainingPoints_gauss_ESM(N, mu, sigma)
        
        if step % step_valid == 0:
            X_valid_torch, s_X_valid_torch = generateTrainingPoints_gauss_ESM(N, mu, sigma)
            lossAdam_valid = loss_ESM(model, X_valid_torch, s_X_valid_torch)
            savedloss_valid.append(lossAdam_valid.item())
            savedloss_valid_idx.append(itera)

            print(
                'Iter %5d, Loss_Train: %.5e, Loss_Valid: %.5e' % (itera, lossAdam.item(), lossAdam_valid.item())
                )
        
        lossAdam.backward(retain_graph = True)

        if step == Adam_iter:
            break

        optimizerAdam.step()
        itera += 1

    print('====================================================')
    print()
    print('Evolution of loss')
    print()
    
    plt.figure(figsize=(6, 2))
    plt.ylim(min(savedloss)/10.0, max(savedloss)*10.0)
    plt.yscale("log")
    plt.plot(savedloss_idx, savedloss, label = "training loss")
    plt.plot(savedloss_valid_idx, savedloss_valid, label = "validation loss")
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.legend()
    plt.show()
    
    # number of test points
    N_test = 2000
    X_test = np.linspace(mu-3.0*sigma, mu+3.0*sigma, N).reshape(N,1)
    X_test_torch = torch.tensor(X_test).float().to(device)
    u_pred = model(X_test_torch).detach().cpu().numpy()
    u_exact = -(X_test - mu)/(sigma**2)
    
    print('solution and errors:')
    print()
    solution_plot(X_test, u_pred, u_exact)
        
    return

def solution_plot(X_test, u_pred, u_exact):
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 1, 1]})
    a0.plot(X_test, u_pred, label = "prediction")
    a0.plot(X_test, u_exact, label = "exact solution")
    a0.set_xlabel('X')
    a0.legend()

    a1.plot(X_test, np.abs(u_pred-u_exact))
    a1.set_xlabel('X')
    a1.set(title='abs. error')

    a2.plot(X_test, np.abs(u_pred-u_exact))
    a2.set_xlabel('X')
    a2.set(title='abs. error')
    a2.set_yscale('log')

    f.tight_layout()
    plt.show()

    return

#------------------------
#------------------------
#------------------------


def generateTrainingPoints_gauss_ISM(N, mu, sigma):
    X = sigma*np.random.randn(N,1) + mu
    X_torch = torch.from_numpy(X).requires_grad_(True).float().to(device)
    return X_torch

def loss_ISM(model, X):
    value1 = torch.sum((model(X))**2, 1)
    tmp = torch.vmap(jacrev(model))(X)
    value2 = 2.0*torch.sum(torch.diagonal(tmp, 0, 1, 2), 1)
    return torch.mean(value1+value2)

def train_model_gauss_ISM(N, mu, sigma):
    
    N_neurons = 10
    model = Plain(1, N_neurons, 1).to(device)
    print('model = ', model)
    print(f"Number of paramters: {count_parameters(model)}")
    print()
    
    Adam_iter = 10000 # stop when iter > Adam_iter
    update_lr_1 = 5000  # change adam lr from 0.01 to 0.001 at update_lr_1 iteration
    update_lr_2 = 8000  # change adam lr from 0.001 to 0.0001 at update_lr_2 iteration

    optimizerAdam = torch.optim.Adam(
        model.parameters(), 
        lr=0.01
    )

    itera = 0

    savedloss = []
    savedloss_idx = []
    savedloss_valid = []
    savedloss_valid_idx = []
    
    X_train_torch = generateTrainingPoints_gauss_ISM(N, mu, sigma)
    
    step_valid = Adam_iter/10
    step_resample = Adam_iter/100
    
    print('====================================================')
    print('Strat training')
    print()
    print('Adam optimizer, re-sample every ', step_resample, ' steps')
    print()
    
    model.train()
    for step in range(Adam_iter+1):
        if step == update_lr_1:
            print('change learning rate to 0.001')
            for g in optimizerAdam.param_groups:
                g['lr'] = 0.001
        if step == update_lr_2:
            print('change learning rate to 0.0001')
            for g in optimizerAdam.param_groups:
                g['lr'] = 0.0001

        # Backward and optimize
        optimizerAdam.zero_grad()

        lossAdam = loss_ISM(model, X_train_torch)

        savedloss.append(lossAdam.item())
        savedloss_idx.append(itera)

        if step % step_resample == 0:
            X_train_torch = generateTrainingPoints_gauss_ISM(N, mu, sigma)
        
        if step % step_valid == 0:
            X_valid_torch = generateTrainingPoints_gauss_ISM(N, mu, sigma)
            lossAdam_valid = loss_ISM(model, X_valid_torch)
            savedloss_valid.append(lossAdam_valid.item())
            savedloss_valid_idx.append(itera)

            print(
                'Iter %5d, Loss_Train: %.5e, Loss_Valid: %.5e' % (itera, lossAdam.item(), lossAdam_valid.item())
                )
        
        lossAdam.backward(retain_graph = True)

        if step == Adam_iter:
            break

        optimizerAdam.step()
        itera += 1

    print('====================================================')
    print()
    print('Evolution of loss')
    print()
    
    plt.figure(figsize=(6, 2))
    plt.plot(savedloss_idx, savedloss, label = "training loss")
    plt.plot(savedloss_valid_idx, savedloss_valid, label = "validation loss")
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.show()
    
    # number of test points
    N_test = 2000
    X_test = np.linspace(mu-3.0*sigma, mu+3.0*sigma, N).reshape(N,1)
    X_test_torch = torch.tensor(X_test).float().to(device)
    u_pred = model(X_test_torch).detach().cpu().numpy()
    u_exact = -(X_test - mu)/(sigma**2)
    
    print('solution and errors:')
    print()
    solution_plot(X_test, u_pred, u_exact)
    
    return

#------------------------
#------------------------
#------------------------

def loss_DSM(model, X, s_X):
    return torch.mean((X[:,1:2]*model(X) - s_X)**2)

def generateTrainingPoints_gauss_DSM(N, mu, sigma):
    X = sigma*np.random.randn(N,2) + mu
    z = np.random.randn(N,1)
    sigma_noise = np.random.rand(N,1)*0.99+0.01
    X[:,0:1] = X[:,0:1] + sigma_noise*z
    X[:,1:2] = sigma_noise
    s_X = -z
    
    X_torch = torch.from_numpy(X).requires_grad_(True).float().to(device)
    s_X_torch = torch.from_numpy(s_X).float().to(device)
    return X_torch, s_X_torch


def train_model_gauss_DSM(N, mu, sigma):
    
    N_neurons = 10
    model = Plain(2, N_neurons, 1).to(device)
    print('model = ', model)
    print(f"Number of paramters: {count_parameters(model)}")
    print()
    
    Adam_iter = 10000 # stop when iter > Adam_iter
    update_lr_1 = 5000  # change adam lr from 0.01 to 0.001 at update_lr_1 iteration
    update_lr_2 = 8000  # change adam lr from 0.001 to 0.0001 at update_lr_2 iteration

    optimizerAdam = torch.optim.Adam(
        model.parameters(), 
        lr=0.01
    )

    itera = 0

    savedloss = []
    savedloss_idx = []
    savedloss_valid = []
    savedloss_valid_idx = []
    
    X_train_torch, s_X_train_torch = generateTrainingPoints_gauss_DSM(N, mu, sigma)
    
    step_valid = Adam_iter/10
    step_resample = Adam_iter/100
    
    print('====================================================')
    print('Strat training')
    print()
    print('Adam optimizer, re-sample every ', step_resample, ' steps')
    print()
    
    model.train()
    for step in range(Adam_iter+1):
        if step == update_lr_1:
            print('change learning rate to 0.001')
            for g in optimizerAdam.param_groups:
                g['lr'] = 0.001
        if step == update_lr_2:
            print('change learning rate to 0.0001')
            for g in optimizerAdam.param_groups:
                g['lr'] = 0.0001

        # Backward and optimize
        optimizerAdam.zero_grad()

        lossAdam = loss_DSM(model, X_train_torch, s_X_train_torch)

        savedloss.append(lossAdam.item())
        savedloss_idx.append(itera)

        if step % step_resample == 0:
            X_train_torch, s_X_train_torch = generateTrainingPoints_gauss_DSM(N, mu, sigma)
        
        if step % step_valid == 0:
            X_valid_torch, s_X_valid_torch = generateTrainingPoints_gauss_DSM(N, mu, sigma)
            lossAdam_valid = loss_DSM(model, X_valid_torch, s_X_valid_torch)
            savedloss_valid.append(lossAdam_valid.item())
            savedloss_valid_idx.append(itera)

            print(
                'Iter %5d, Loss_Res: %.5e, Loss_Valid: %.5e' % (itera, lossAdam.item(), lossAdam_valid.item())
                )
        
        lossAdam.backward(retain_graph = True)

        if step == Adam_iter:
            break

        optimizerAdam.step()
        itera += 1

    print('====================================================')
    print()
    print('Evolution of loss')
    print()
    
    plt.figure(figsize=(6, 2))
    plt.plot(savedloss_idx, savedloss, label = "training loss")
    plt.plot(savedloss_valid_idx, savedloss_valid, label = "validation loss")
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.legend()
    plt.show()
    
    print('solution and errors:')
    print()
    
    # number of test points
    N_test = 2000
    X_test = np.linspace(mu-3.0*sigma, mu+3.0*sigma, N).reshape(N,1)
    t = 0.0
    print('t = ', t)
    X_test_torch = torch.tensor(np.hstack([X_test, 0.0*X_test+t])).float().to(device)
    u_pred = model(X_test_torch).detach().cpu().numpy()
    u_exact = -(X_test - mu)/(sigma**2+t**2)
    
    solution_plot(X_test, u_pred, u_exact)
    
    t = 0.1
    print('t = ', t)
    X_test_torch = torch.tensor(np.hstack([X_test, 0.0*X_test+t])).float().to(device)
    u_pred = model(X_test_torch).detach().cpu().numpy()
    u_exact = -(X_test - mu)/(sigma**2+t**2)
    
    solution_plot(X_test, u_pred, u_exact)
    
    t = 1.0
    print('t = ', t)
    X_test_torch = torch.tensor(np.hstack([X_test, 0.0*X_test+t])).float().to(device)
    u_pred = model(X_test_torch).detach().cpu().numpy()
    u_exact = -(X_test - mu)/(sigma**2+t**2)
    
    solution_plot(X_test, u_pred, u_exact)

    X1 = np.linspace(mu-3.0*sigma-1.0, mu+3.0*sigma+1.0, N)
    T2 = np.linspace(0, 1, 100)
    Y1, Y2 = np.meshgrid(X1, T2)
    y1_tst = Y1.reshape(100*N, 1)
    y2_tst = Y2.reshape(100*N, 1)
    Y_tst = np.hstack((y1_tst, y2_tst))
    T_tst_torch = torch.tensor(Y_tst).float().to(device)
    u_pred = model(T_tst_torch).detach().cpu().numpy()
    u_pred = u_pred.reshape(100, N)
    u_exact = -(Y1 - mu)/(sigma**2+Y2**2)
    
    #fig = plt.figure(figsize=(10, 3))
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 1, 1]})
    #a0 = fig.add_subplot(1, 3, 1, projection='3d')
    a0.contourf(Y1, Y2, u_pred, cmap = cm.coolwarm)
    a0.set_xlabel('X')
    a0.set_ylabel('t')
    a0.set_title('s_{approx}(X, t)')
    
    #a1 = fig.add_subplot(1, 3, 2, projection='3d')
    a1.contourf(Y1, Y2, u_exact, cmap = cm.gist_earth)
    a1.set_xlabel('X')
    a1.set_ylabel('t')
    a1.set_title('s_{exact}(X, t)')
        
    X_tst_torch, s_X_tst_torch = generateTrainingPoints_gauss_DSM(1000, mu, sigma)
    u_pred2 = model(X_tst_torch).detach().cpu().numpy()
    X_tst = X_tst_torch.detach().cpu().numpy()
    
    a2.scatter(X_tst[:,0], X_tst[:,1], c=u_pred2, cmap = cm.gist_earth)
    a2.set_xlabel('X')
    a2.set_ylabel('t')
    a2.set_title('s(X, t)')
    
    plt.show()
    
    return

#------------------------
#------------------------
#------------------------

def loss_DSM2(model, X, s_X, t):
    return torch.mean((t*model(X) - s_X)**2)

def generateTrainingPoints_DSM(N, mu, sigma, t):
    z = np.random.randn(N,1)
    X = sigma*np.random.randn(N,1) + mu + t*z
    X_torch = torch.from_numpy(X).requires_grad_(True).float().to(device)
    s_X = -z
    s_X_torch = torch.from_numpy(s_X).float().to(device)
    return X_torch, s_X_torch

def train_model_DSM(N, mu, sigma, t):
    
    N_neurons = 10
    model = Plain(1, N_neurons, 1).to(device)
    print('model = ', model)
    print(f"Number of paramters: {count_parameters(model)}")
    print()
    
    Adam_iter = 10000 # stop when iter > Adam_iter
    update_lr_1 = 5000  # change adam lr from 0.01 to 0.001 at update_lr_1 iteration
    update_lr_2 = 8000  # change adam lr from 0.001 to 0.0001 at update_lr_2 iteration

    optimizerAdam = torch.optim.Adam(
        model.parameters(), 
        lr=0.01
    )

    itera = 0

    savedloss = []
    savedloss_idx = []
    savedloss_valid = []
    savedloss_valid_idx = []
    
    X_train_torch, s_X_train_torch = generateTrainingPoints_DSM(N, mu, sigma, t)
    
    step_valid = Adam_iter/10
    step_resample = Adam_iter/100
    
    print('====================================================')
    print('Strat training')
    print()
    print('Adam optimizer, re-sample every ', step_resample, ' steps')
    print()
    
    model.train()
    for step in range(Adam_iter+1):
        if step == update_lr_1:
            print('change learning rate to 0.001')
            for g in optimizerAdam.param_groups:
                g['lr'] = 0.001
        if step == update_lr_2:
            print('change learning rate to 0.0001')
            for g in optimizerAdam.param_groups:
                g['lr'] = 0.0001

        # Backward and optimize
        optimizerAdam.zero_grad()

        lossAdam = loss_DSM2(model, X_train_torch, s_X_train_torch, t)

        savedloss.append(lossAdam.item())
        savedloss_idx.append(itera)

        if step % step_resample == 0:
            X_train_torch, s_X_train_torch = generateTrainingPoints_DSM(N, mu, sigma, t)
        
        if step % step_valid == 0:
            X_valid_torch, s_X_valid_torch = generateTrainingPoints_DSM(N, mu, sigma, t)
            lossAdam_valid = loss_DSM2(model, X_valid_torch, s_X_valid_torch, t)
            savedloss_valid.append(lossAdam_valid.item())
            savedloss_valid_idx.append(itera)

            print(
                'Iter %5d, Loss_Res: %.5e, Loss_Valid: %.5e' % (itera, lossAdam.item(), lossAdam_valid.item())
                )
        
        lossAdam.backward(retain_graph = True)

        if step == Adam_iter:
            break

        optimizerAdam.step()
        itera += 1

    print('====================================================')
    print()
    print('Evolution of loss')
    print()
    
    plt.figure(figsize=(6, 2))
    plt.ylim(min(savedloss)/10.0, max(savedloss)*10.0)
    plt.yscale("log")
    plt.plot(savedloss_idx, savedloss, label = "training loss")
    plt.plot(savedloss_valid_idx, savedloss_valid, label = "validation loss")
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.legend()
    plt.show()
    
    # number of test points
    N_test = 2000
    X_test = np.linspace(mu-3.0*sigma, mu+3.0*sigma, N).reshape(N,1)
    X_test_torch = torch.tensor(X_test).float().to(device)
    u_pred = model(X_test_torch).detach().cpu().numpy()
    u_exact = -(X_test - mu)/(sigma**2)
    
    print('solution and errors:')
    print()
    
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 1, 1]})
    a0.plot(X_test, u_pred, label = "s_t(X)")
    a0.plot(X_test, u_exact, label = "s_0(X)")
    a0.set_xlabel('X')
    a0.legend()

    a1.plot(X_test, np.abs(u_pred-u_exact))
    a1.set_xlabel('X')
    a1.set(title='abs. error')

    a2.plot(X_test, np.abs(u_pred-u_exact))
    a2.set_xlabel('X')
    a2.set(title='abs. error')
    a2.set_yscale('log')

    f.tight_layout()
    plt.show()
    
    return