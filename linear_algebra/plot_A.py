import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

def plot_A(A):
    O1 = np.array([[0.], [0.]])

    f, (a0, a1, a2, a3, a4) = plt.subplots(1, 5, figsize=(15, 5))

    theta = 0*np.pi/180
    E1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    E2 = np.array([[np.cos(theta+np.pi/2)], [np.sin(theta+np.pi/2)]])
    b = np.hstack((O1, E1, E1+E2, E2))
    a0.plot(b[0, [0, 1]], b[1, [0, 1]], 'b')
    a0.plot(b[0, [0, 3]], b[1, [0, 3]], 'r')
    X = A@b
    a0.fill(b[0,:], b[1,:], "c", label='v')
    a0.fill(X[0,:], X[1,:], "tab:blue", label='Av', alpha=.8)
    a0.plot(X[0, [0, 1]], X[1, [0, 1]], 'b', linewidth=4)
    a0.plot(X[0, [0, 3]], X[1, [0, 3]], 'r', linewidth=4)
    a0.set_aspect('equal')
    a0.legend(loc='upper right')

    theta = 45*np.pi/180
    E1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    E2 = np.array([[np.cos(theta+np.pi/2)], [np.sin(theta+np.pi/2)]])
    b = np.hstack((O1, E1, E1+E2, E2))
    a1.plot(b[0, [0, 1]], b[1, [0, 1]], 'b')
    a1.plot(b[0, [0, 3]], b[1, [0, 3]], 'r')
    X = A@b
    a1.fill(b[0,:], b[1,:], "c", label='v')
    a1.fill(X[0,:], X[1,:], "tab:blue", label='Av', alpha=.8)
    a1.plot(X[0, [0, 1]], X[1, [0, 1]], 'b', linewidth=4)
    a1.plot(X[0, [0, 3]], X[1, [0, 3]], 'r', linewidth=4)
    a1.set_aspect('equal')
    a1.legend(loc='upper right')

    theta = 90*np.pi/180
    E1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    E2 = np.array([[np.cos(theta+np.pi/2)], [np.sin(theta+np.pi/2)]])
    b = np.hstack((O1, E1, E1+E2, E2))
    a2.plot(b[0, [0, 1]], b[1, [0, 1]], 'b')
    a2.plot(b[0, [0, 3]], b[1, [0, 3]], 'r')
    X = A@b
    a2.fill(b[0,:], b[1,:], "c", label='v')
    a2.fill(X[0,:], X[1,:], "tab:blue", label='Av', alpha=.8)
    a2.plot(X[0, [0, 1]], X[1, [0, 1]], 'b', linewidth=4)
    a2.plot(X[0, [0, 3]], X[1, [0, 3]], 'r', linewidth=4)
    a2.set_aspect('equal')
    a2.legend(loc='upper right')

    theta = 135*np.pi/180
    E1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    E2 = np.array([[np.cos(theta+np.pi/2)], [np.sin(theta+np.pi/2)]])
    b = np.hstack((O1, E1, E1+E2, E2))
    a3.plot(b[0, [0, 1]], b[1, [0, 1]], 'b')
    a3.plot(b[0, [0, 3]], b[1, [0, 3]], 'r')
    X = A@b
    a3.fill(b[0,:], b[1,:], "c", label='v')
    a3.fill(X[0,:], X[1,:], "tab:blue", label='Av', alpha=.8)
    a3.plot(X[0, [0, 1]], X[1, [0, 1]], 'b', linewidth=4)
    a3.plot(X[0, [0, 3]], X[1, [0, 3]], 'r', linewidth=4)
    a3.set_aspect('equal')
    a3.legend(loc='upper right')

    theta = 180*np.pi/180
    E1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    E2 = np.array([[np.cos(theta+np.pi/2)], [np.sin(theta+np.pi/2)]])
    b = np.hstack((O1, E1, E1+E2, E2))
    a4.plot(b[0, [0, 1]], b[1, [0, 1]], 'b')
    a4.plot(b[0, [0, 3]], b[1, [0, 3]], 'r')
    X = A@b
    a4.fill(b[0,:], b[1,:], "c", label='v')
    a4.fill(X[0,:], X[1,:], "tab:blue", label='Av', alpha=.8)
    a4.plot(X[0, [0, 1]], X[1, [0, 1]], 'b', linewidth=4)
    a4.plot(X[0, [0, 3]], X[1, [0, 3]], 'r', linewidth=4)
    a4.set_aspect('equal')
    a4.legend(loc='upper right')

    f.tight_layout()

    plt.show()
    
def plot_A_abs(A):
    A2 = np.transpose(A) @ A
    eigenvalues, eigenvectors = linalg.eig(A2)
    A_abs = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ linalg.inv(eigenvectors)
    
    O1 = np.array([[0.], [0.]])

    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10, 3))

    O1 = np.array([[0.], [0.]])
    E1 = eigenvectors[:,0:1]
    E2 = eigenvectors[:,1:2]
    b = np.hstack((O1, E1, E1+E2, E2))
    a0.plot(b[0, [0, 1]], b[1, [0, 1]], 'b')
    a0.plot(b[0, [0, 3]], b[1, [0, 3]], 'r')
    X = A@b
    a0.fill(b[0,:], b[1,:], "c", label='v')
    a0.fill(X[0,:], X[1,:], "tab:blue", label='Av', alpha=.8)
    a0.plot(X[0, [0, 1]], X[1, [0, 1]], 'b', linewidth=4)
    a0.plot(X[0, [0, 3]], X[1, [0, 3]], 'r', linewidth=4)
    a0.set_aspect('equal')
    a0.legend(loc='upper right')

    a1.plot(b[0, [0, 1]], b[1, [0, 1]], 'b')
    a1.plot(b[0, [0, 3]], b[1, [0, 3]], 'r')
    X = A_abs@b
    a1.fill(b[0,:], b[1,:], "c", label='v')
    a1.fill(X[0,:], X[1,:], "tab:blue", label='|A|v', alpha=.8)
    a1.plot(X[0, [0, 1]], X[1, [0, 1]], 'b', linewidth=4)
    a1.plot(X[0, [0, 3]], X[1, [0, 3]], 'r', linewidth=4)
    a1.set_aspect('equal')
    a1.legend(loc='upper right')

    theta = np.linspace(0, 1)*np.pi*3.0/2.0+np.pi/2.0
    x = np.cos(theta).reshape(1, 50)
    y = np.sin(theta).reshape(1, 50)
    v = np.vstack((x, y))
    av = A@v    
    a2.plot(v[0,:], v[1,:], 'b', label='v')
    a2.plot(av[0,:], av[1,:], 'r', label='Av')
    
    theta = np.linspace(0, 1)*np.pi/2.0
    x = np.cos(theta).reshape(1, 50)
    y = np.sin(theta).reshape(1, 50)
    v = np.vstack((x, y))
    av = A@v    
    a2.plot(v[0,:], v[1,:], 'b:')
    a2.plot(av[0,:], av[1,:], 'r:')
    a2.set_aspect('equal')
    a2.legend(loc='upper right')
    
    f.tight_layout()
    plt.show()
    