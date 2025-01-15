import numpy as np
from scipy.sparse.linalg import svds
from math import sqrt
import time
import matplotlib.pyplot as plt

def SVD_initialization(X, r1, r2):
    M = np.sqrt(np.abs(X))
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    A = U[:, :r1]
    B = np.diag(S[:r1]) @ Vt[:r1, :]

    M = M*np.sign(X)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    C = U[:, :r2]
    D = np.diag(S[:r2]) @ Vt[:r2, :]

    return A, B, C, D

def initialization_multiple(X, r, n_factors):
    factors = {}
    target = X.copy()
    for i in range(1, n_factors):
        r1 = r
        r2 = r*(n_factors-i)

        A, B, C, D = SVD_initialization(target, r1, r2)

        factors[i-1] = (A, B)
        target = C@D
    factors[n_factors-1] = (C, D)

    return factors

def beta_mometum(dec, beta, beta_b, g, g_b, eta):
    '''beta parameter computation for the accelerated method with momentum'''
    if dec:
        beta = min(beta_b, g*beta)
        beta_b = min(1, g_b*beta_b)
    else:
        beta_b = beta
        beta /= eta

    return beta, beta_b

def hadamardLS(s, A, b, x):
    '''Least square problem analytic resolution'''
    H = A.T@np.diag(s**2)@A
    d = A.T@(s*b)
    try:
        x = np.linalg.solve(H, d)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(H) @ d

    return x

def updateH(WH1, W2, H2, X, beta):
    '''Column-by-column least square resolution with momentum acceleration'''
    m, n = X.shape

    H2_old = H2.copy()
    for j in range(n):
        H2[:,j] = hadamardLS(WH1[:,j], W2, X[:,j], H2[:,j])

    H2 += beta * (H2 - H2_old)

    return H2, H2_old


def BCD_multiple(X, r=0, n_factors=0, initial_factors=[], maxiter=100, beta=0.5, g=1.05, g_b=1.01, eta=1.5):
    ''' Block-Coordinate Descent with spectral initialization for Hadamard decomposition with k rank-r factors 
    
    Params:
    X: input matrix (numpy array) 
    r: rank of the loxw-rank factors if not provided (integer) 
    n_factors : number of low-rank factors if not provided (float)
    initial_factors : list of initial low-rank factors (python list)
    maxiter : maximum number of iteration to perform (integer)
    beta : hyperparameter for momentum accelertion
    g : hyperparameter for momentum accelertion
    g_b : hyperparameter for momentum accelertion
    eta : hyperparameter for momentum accelertion
    
    Returns: 
    factors : final factors of the decomposition
    error : approximation error at each iteration
    times : computation time for each iteration
    ''' 
    if not initial_factors:
        factors = initialization_multiple(X, r, n_factors)
    else:
        factors = {}
        n_factors = len(initial_factors)
        for i in range(n_factors):
            factors[i] = initial_factors[i]

    beta = 0.5
    beta_b = 1
    c=0

    old_factors = {}

    X_approx = np.prod([W@H for (W, H) in factors.values()], axis=0)    
    error = [np.linalg.norm(X-X_approx, "fro")/np.linalg.norm(X, "fro")]
    times = [0]
    t0 = time.time()

    for it in range(maxiter):
        for k in range(n_factors):
            W2, H2 = factors[k]
            WH1 = np.prod([W@H for i, (W, H) in factors.items() if i != k], axis=0)

            H2, H2_old = updateH(WH1, W2, H2, X, beta)

            W2, W2_old = updateH(WH1.T, H2.T, W2.T, X.T, beta)
            W2, W2_old = W2.T, W2_old.T

            factors[k] = (W2.copy(), H2.copy())
            old_factors[k] = (W2_old.copy(), H2_old.copy())

        X_approx = np.prod([W@H for (W, H) in factors.values()], axis=0)
        
        err = np.linalg.norm(X-X_approx, "fro")/np.linalg.norm(X, "fro")
        times.append(time.time()-t0)

        dec = err < error[-1]
        if not dec: factors = old_factors.copy()
        beta, beta_b = beta_mometum(dec, beta, beta_b, g, g_b, eta)
        
        if 1 - err/error[-1] <= 10**(-8): c+=1
        else: c=0
        error.append(err)
        if c>10 or err<10**(-10): break

    return factors, error, times