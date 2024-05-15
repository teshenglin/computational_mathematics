import numpy as np
import matplotlib.pyplot as plt
from random import sample 

def SDE_plot_1(N, M, T, theo_mean, theo_std, time, Xh_0):
    # Maximum likelihood estimation
    mean = np.mean(Xh_0[:,M])
    std = np.std(Xh_0[:,M])
    y = np.linspace(-10, 10, 100)
    height_M = 1.0/std/np.sqrt(2.0*np.pi)
    pdf = np.exp(-0.5*((y-mean)/std)**2)/std/np.sqrt(2.0*np.pi)
    
    print('The right panel shows the results at X(t=T) and its maximum likelihood estimation (MLE).')
    print('MLE:')
    print('meam = ', mean)
    print('std = ', std)
    print('')
    print('Theoretical results')
    print('Theoretical meam = ', theo_mean)
    print('Theoretical std = ', theo_std)
    print('')

    f, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [10, 1]})
    for ii in sample(range(N), 20):
        a0.plot(time, Xh_0[ii,:])

    imax = np.argmax(Xh_0[:,M])
    a0.plot(time, Xh_0[imax,:])
    imin = np.argmin(Xh_0[:,M])
    a0.plot(time, Xh_0[imin,:])
    imax = np.argmax(Xh_0[:,0])
    a0.plot(time, Xh_0[imax,:])
    imin = np.argmin(Xh_0[:,0])
    a0.plot(time, Xh_0[imin,:])

    a0.set_xlim(0, T)
    a0.set_ylim(-10, 10)
    a0.set(xlabel='t')
    a0.set(title='X(t)')

    a1.scatter(Xh_0[:,M]*0-0.2, Xh_0[:,M], s=10.0)
    a1.plot(pdf, y)
    a1.set_xlim(-height_M*2, height_M*2)
    a1.set_ylim(-10, 10)
    a1.set_xticklabels([])
    a1.set_yticklabels([])

    f.tight_layout()
    plt.show()
    