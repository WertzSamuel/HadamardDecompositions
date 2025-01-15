import numpy as np
from scipy.sparse.linalg import svds
from math import sqrt, pow
import time

def scaling(X, A, B, C, D):
    F = (A@B)*(C@D)
    num = np.sum(F * X)
    deno = np.linalg.norm(F, 'fro')**2
    frac = num/deno
    alpha = pow(abs(frac), 1/4)

    if frac<0: return A*(-alpha), B*alpha, C*alpha, D*alpha
    return A*alpha, B*alpha, C*alpha, D*alpha

def rnd_initialization(X, m, n, r):
    A = np.random.randn(m,r)
    B = np.random.randn(r,n)
    C = np.random.randn(m,r)
    D = np.random.randn(r,n)

    return A, B, C, D

def uniform_Xavier(X, m, n, r):
    mean = sqrt(6/(m+n))
    A = np.random.uniform(-mean, mean, (m,r))
    B = np.random.uniform(-mean, mean, (r,n))
    C = np.random.uniform(-mean, mean, (m,r))
    D = np.random.uniform(-mean, mean, (r,n))

    return A, B, C, D

def normal_Xavier(X, m, n, r):
    deviation = sqrt(2/(m+n))
    A = np.random.normal(loc=0, scale=deviation, size=(m,r))
    B = np.random.normal(loc=0, scale=deviation, size=(r,n))
    C = np.random.normal(loc=0, scale=deviation, size=(m,r))
    D = np.random.normal(loc=0, scale=deviation, size=(r,n))

    return A, B, C, D

def SVD_initialization(X, m, n, r):
    M = np.sqrt(np.abs(X))
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    A = U[:, :r]
    B = np.diag(S[:r]) @ Vt[:r, :]

    M = M*np.sign(X)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    C = U[:, :r]
    D = np.diag(S[:r]) @ Vt[:r, :]

    return A, B, C, D

from sklearn.cluster import KMeans
def kmeans_initialization(X, m, n, r):
    kmeans = KMeans(n_clusters=r, random_state=42)

    M = np.sqrt(np.abs(X))
    kmeans.fit(M.T)
    A = kmeans.cluster_centers_.T

    labels = kmeans.labels_
    n = X.shape[1]
    B = np.zeros((r, n))
    B[labels, np.arange(n)] = 1

    M = M*np.sign(X)
    kmeans.fit(M.T)
    C = kmeans.cluster_centers_.T

    labels = kmeans.labels_
    n = X.shape[1]
    D = np.zeros((r, n))
    D[labels, np.arange(n)] = 1

    return A, B, C, D

def hadamardLS_optimal(s, A, b, x, maxiter):
    '''Least square problem GD with optimal step-size resolution'''
    H = A.T@np.diag(s**2)@A
    d = A.T@(s*b)

    for k in range(maxiter):
        gradient = H@x - d
        deno = gradient.T@H@gradient
        if deno<10**(-6): break
        lr = np.linalg.norm(gradient)**2 / deno
        x = x -lr*gradient

    return x

def hadamardLS_lipschitz(s, A, b, x, maxiter):
    '''Least square problem GD with lipschitz step-size resolution'''
    H = A.T@np.diag(s**2)@A
    d = A.T@(s*b)
    L = svds(np.diag(s)@A, k=1, return_singular_vectors=False)[0]**2
    lr = 1/L
    for k in range(maxiter):
        gradient = H@x - d
        x = x - lr*gradient

    return x

def hadamardLS_exact(s, A, b, x, maxiter):
    '''Least square problem analytic resolution'''
    H = A.T@np.diag(s**2)@A
    d = A.T@(s*b)
    try:
        x = np.linalg.solve(H, d)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(H) @ d
    return x

def updateH(W1, H1, W2, H2, X, alpha, hadamardLS, beta):
    '''Column-by-column least square resolution'''
    m, n = X.shape
    r = len(H2)
    opt_k = 1 + int(alpha*(m*n + m*r)/(r*n))

    for j in range(n):
        H2[:,j] = hadamardLS(W1@H1[:,j], W2, X[:,j], H2[:,j], opt_k)

    return H2

def updateH_momentum(W1, H1, W2, H2, X, alpha, hadamardLS, beta):
    '''Column-by-column least square resolution with momentum acceleration'''
    m, n = X.shape
    r = len(H2)
    opt_k = 1 + int(alpha*(m*n + m*r)/(r*n))

    H2_old = H2.copy()
    for j in range(n):
        H2[:,j] = hadamardLS(W1@H1[:,j], W2, X[:,j], H2[:,j], opt_k)

    H2 += beta * (H2 - H2_old)

    return H2, H2_old

def beta_mometum(dec, beta, beta_b, g, g_b, eta):
    '''beta parameter computation for the accelerated method with momentum'''
    if dec:
        beta = min(beta_b, g*beta)
        beta_b = min(1, g_b*beta_b)
    else:
        beta_b = beta
        beta /= eta

    return beta, beta_b


def Hadamard_BCD(X, r, beta=0.5, g=1.05, g_b=1.01, eta=1.5, alpha=0, maxiter=100, initialization=SVD_initialization, hadamardLS=hadamardLS_exact, update=updateH_momentum):
    ''' Block-Coordinate Descent for Hadamard decomposition with 2 rank-r factors 
    
    Params:
    X: input matrix (numpy array) 
    r: rank of the low-rank factors if not provided (integer) 
    maxiter : maximum number of iteration to perform (integer)
    initialization: method to initialize W1, H1, W2 and H2 (rnd_initialization, uniform_Xavier, normal_xavier, SVD_initialization or kmeans_initialization)
    hadamardLS: least square resolution method to use (hadamardLS_optimal, hadamardLS_lipschitz or hadamardLS_exact)
    update: update method to use -> normal (updateH) or accelerated (updateH_momentum)
    beta : hyperparameter for momentum accelertion
    g : hyperparameter for momentum accelertion
    g_b : hyperparameter for momentum accelertion
    eta : hyperparameter for momentum accelertion
    
    Returns: 
    factors : final factors of the decomposition
    error : approximation error at each iteration
    times : computation time for each iteration
    ''' 
    m, n = X.shape
    W1, H1, W2, H2 = initialization(X, m, n, r)
    W1, H1, W2, H2 = scaling(X, W1, H1, W2, H2)
    #[W1, H1, W2, H2] = np.load("init_SVD.npy")
    #H1, H2 = H1.T, H2.T
    #np.save("init_SVD.npy", np.array([W1, H1.T, W2, H2.T]))
    beta = 0.5
    beta_b = 1

    c=0

    error = [np.linalg.norm(X-(W1@H1)*(W2@H2), "fro")/np.linalg.norm(X, "fro")]
    times = [0]
    t0 = time.time()

    for k in range(maxiter):
        if time.time()-t0 > 200 : break
        H2, H2_old = update(W1, H1, W2, H2, X, alpha, hadamardLS, beta)
        W2, W2_old = update(H1.T, W1.T, H2.T, W2.T, X.T, alpha, hadamardLS, beta)
        W2, W2_old = W2.T, W2_old.T
        H1, H1_old = update(W2, H2, W1, H1, X, alpha, hadamardLS, beta)
        W1, W1_old = update(H2.T, W2.T, H1.T, W1.T, X.T, alpha, hadamardLS, beta)
        W1, W1_old = W1.T, W1_old.T

        err = np.linalg.norm(X-(W1@H1)*(W2@H2), "fro")/np.linalg.norm(X, "fro")
        times.append(time.time()-t0)

        dec = err < error[-1]
        if not dec: 
            H2 -= beta * (H2 - H2_old)
            W2 -= beta * (W2 - W2_old)
            H1 -= beta * (H1 - H1_old)
            W1 -= beta * (W1 - W1_old)
            err = np.linalg.norm(X-(W1@H1)*(W2@H2), "fro")/np.linalg.norm(X, "fro")
        beta, beta_b = beta_mometum(dec, beta, beta_b, g, g_b, eta)
        
        if 1 - err/error[-1] <= 10**(-8): c+=1
        else: c=0
        error.append(err)
        if c>10 or err<10**(-10): break


    return W1, H1, W2, H2, error, times