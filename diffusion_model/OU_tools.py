import numpy as np
import matplotlib.pyplot as plt
from random import sample 

def OU_forward(beta, sigma, mu_0, sigma_0, theo_mean, theo_std, T, N, M):
    dt = T/M
    time = np.linspace(0, T, M+1)

    # Euler-Maruyama method:
    X_0 = np.zeros((N, M+1))

    # sample initial condition from N(0, 1)
    X_0[:,0] = np.random.randn(N)*sigma_0 + mu_0

    # iteration
    for ii in range(M):
        X_0[:, ii+1] = X_0[:, ii] - beta*dt*X_0[:, ii] + sigma*np.sqrt(dt)*np.random.randn(N)

    # maximum likelihood estimation at t= T
    mean_M = np.mean(X_0[:,M])
    std_M = np.std(X_0[:,M])

    # The corresponding normal distribution
    y = np.linspace(-10, 10, 100)

    # at t=0
    height_0 = 1.0/sigma_0/np.sqrt(2.0*np.pi)
    pdf_0 = np.exp(-0.5*((y-mu_0)/sigma_0)**2)*height_0

    # at t=T
    height_M = 1.0/std_M/np.sqrt(2.0*np.pi)
    pdf_M = np.exp(-0.5*((y-mean_M)/std_M)**2)*height_M

    print('The left panel shows the initial sampling at X(t=0).')
    print('')
    print('The right panel shows the results at X(t=T) and its maximum likelihood estimation (MLE).')
    print('MLE:')
    print('meam = ', mean_M)
    print('std = ', std_M)
    print('')
    print('Theoretical results:')
    print('Theoretical meam = ', theo_mean)
    print('Theoretical std = ', theo_std)

    f, (a2, a0, a1) = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 10, 1]})
    for ii in sample(range(N), 20):
        a0.plot(time, X_0[ii,:])

    imax = np.argmax(X_0[:,M])
    a0.plot(time, X_0[imax,:])
    imin = np.argmin(X_0[:,M])
    a0.plot(time, X_0[imin,:])
    imax = np.argmax(X_0[:,0])
    a0.plot(time, X_0[imax,:])
    imin = np.argmin(X_0[:,0])
    a0.plot(time, X_0[imin,:])


    a2.scatter(X_0[:,0]*0-height_0, X_0[:,0], s=10.0)
    a2.plot(pdf_0, y)
    a2.set_xlim(-height_0*2, height_0*2)
    a2.set_ylim(-10, 10)
    a2.set_xticklabels([])
    a2.set_yticklabels([])

    a0.set_xlim(0, T)
    a0.set_ylim(-10, 10)
    a0.set(xlabel='t')
    a0.set(title='X(t)')

    a1.scatter(X_0[:,M]*0-height_M, X_0[:,M], s=10.0)
    a1.plot(pdf_M, y)
    a1.set_xlim(-height_M*2, height_M*2)
    a1.set_ylim(-10, 10)
    a1.set_xticklabels([])
    a1.set_yticklabels([])

    f.tight_layout()
    plt.show()
    
def OU_reverse(beta, sigma, mu_0, sigma_0, mean_T, sigma_T, T, N, M):
    dt = T/M
    time = np.linspace(0, T, M+1)

    # Euler-Maruyama method:
    Y_0 = np.zeros((N, M+1))

    # sample initial condition from N(0, 1)
    Y_0[:,M] = sigma_T*np.random.randn(N)+mean_T

    # iteration
    for ii in range(M):
        tk = (M-ii)*dt
        sigma_tk_square = np.exp(-2.0*beta*tk)*(sigma_0**2)+ (1-np.exp(-2.0*beta*tk))*(sigma**2)/(2.0*beta)
        mt = np.exp(-beta*tk)*mu_0
        score = -(Y_0[:, M-ii] - mt)/sigma_tk_square
        Y_0[:, M-ii-1] = Y_0[:, M-ii] + (-dt)*( -beta*Y_0[:, M-ii] - (sigma**2)*score)
        Y_0[:, M-ii-1] = Y_0[:, M-ii-1] + sigma*np.sqrt(dt)*np.random.randn(N)

    # maximum likelihood estimation at t= 0
    mean_0 = np.mean(Y_0[:, 0])
    std_0 = np.std(Y_0[:, 0])
    
    # The corresponding normal distribution
    y = np.linspace(-10, 10, 100)

    # at t=0
    height_0 = 1.0/std_0/np.sqrt(2.0*np.pi)
    pdf_0 = np.exp(-0.5*((y-mean_0)**2))*height_0

    # at t=T
    height_M = 1.0/sigma_T/np.sqrt(2.0*np.pi)
    pdf_M = np.exp(-0.5*(y/sigma_T)**2)*height_M

    print('The right panel shows the initial sampling at X(t=T).')
    print('')

    print('The left panel shows the results at X(t=0) and its maximum likelihood estimation (MLE).')
    print('MLE:')
    print('meam = ', mean_0)
    print('std = ', std_0)
    print('')
    print('Theoretical results')
    print('Theoretical meam = ', mu_0)
    print('Theoretical std = ', sigma_0)

    f, (a2, a0, a1) = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 10, 1]})
    for ii in sample(range(N), 20):
        a0.plot(time, Y_0[ii,:])

    imax = np.argmax(Y_0[:,0])
    a0.plot(time, Y_0[imax,:])
    imin = np.argmin(Y_0[:,0])
    a0.plot(time, Y_0[imin,:])
    imax = np.argmax(Y_0[:,M])
    a0.plot(time, Y_0[imax,:])
    imin = np.argmin(Y_0[:,M])
    a0.plot(time, Y_0[imin,:])

    a2.scatter(Y_0[:,0]*0-height_0, Y_0[:,0], s=10.0)
    a2.plot(pdf_0, y)
    a2.set_xlim(-height_0*2, height_0*2)
    a2.set_ylim(-10, 10)
    a2.set_xticklabels([])
    a2.set_yticklabels([])

    a0.set_xlim(0, T)
    a0.set_ylim(-10, 10)
    a0.set(xlabel='t')
    a0.set(title='X(t)')

    a1.scatter(Y_0[:,M]*0-height_M, Y_0[:,M], s=10.0)
    a1.plot(pdf_M, y)
    a1.set_xlim(-height_M*2, height_M*2)
    a1.set_ylim(-10, 10)
    a1.set_xticklabels([])
    a1.set_yticklabels([])

    f.tight_layout()
    plt.show()

def OU_reverse_ODE(beta, sigma, mu_0, sigma_0, mean_inf, sigma_inf, T, N, M):
    dt = T/M
    time = np.linspace(0, T, M+1)
    
    N=20

    # Euler-Maruyama method:
    Y_0 = np.zeros((N, M+1))

    # sample initial condition from N(0, 1)
    Y_0[:,M] = np.linspace(-5, 5, N)
    #Y_0[:,M] = sigma_inf*np.random.randn(N)

    # iteration
    for ii in range(M):
        tk = (M-ii)*dt
        sigma_tk_square = np.exp(-2.0*beta*tk)*(sigma_0**2)+ (1-np.exp(-2.0*beta*tk))*(sigma**2)/(2.0*beta)
        mt = np.exp(-beta*tk)*mu_0
        score = -(Y_0[:, M-ii] - mt)/sigma_tk_square
        Y_0[:, M-ii-1] = Y_0[:, M-ii] + (-dt)*( -beta*Y_0[:, M-ii] - sigma**2*score/2.0) 

    f, a0 = plt.subplots(1, 1, figsize=(10, 3))
    for ii in range(N):
        a0.plot(time, Y_0[ii,:])

    imax = np.argmax(Y_0[:,0])
    a0.plot(time, Y_0[imax,:])
    imin = np.argmin(Y_0[:,0])
    a0.plot(time, Y_0[imin,:])
    imax = np.argmax(Y_0[:,M])
    a0.plot(time, Y_0[imax,:])
    imin = np.argmin(Y_0[:,M])
    a0.plot(time, Y_0[imin,:])

    a0.set_xlim(0, T)
    a0.set_ylim(-10, 10)
    a0.set(xlabel='t')
    a0.set(title='X(t)')

    f.tight_layout()
    plt.show()