import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors

####### gaussian mixture & samples #######

def generate_list_rho(n_rho = 3, D = 3):
    rho_list = []
    for i in range(n_rho):
        mu1 = np.random.rand(D)
        sigma_tmp = np.random.rand(D,D)
        sigma1 = 0.01 *((sigma_tmp.T + sigma_tmp)+ D * np.eye(D))
        rho = multivariate_normal(mean = mu1,cov = sigma1)
        rho_list.append(rho)
    return rho_list

def sample_rho_batch(rho_list, nsamples, D=3):
    sample = np.zeros([nsamples,D])
    n_rho = len(rho_list)
    for i in range(nsamples):
        rand = np.random.rand(1)
        idx = int(np.floor(n_rho * rand))
        sample[i,:] = rho_list[idx].rvs()
    return sample

def sample_rho(rho_list):
    n_rho = len(rho_list)
    rand = np.random.rand(1)
    idx = int(np.floor(n_rho * rand))
    sample = rho_list[idx].rvs()
    return sample

####### gradients #######

def grad_h_0(v,X_source,X_target,nu):
    n_target = len(X_target)
    Xv = np.c_[X_target,np.sqrt(-v-np.min(-v))]
    kdTree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Xv)
    Y = X_source
    Yv = np.reshape(np.hstack([Y,0]),[1,-1])
    neighbors = kdTree.kneighbors(Yv,n_neighbors=1,return_distance=False)
    area_vect = np.zeros(n_target)
    area_vect[neighbors[0]] += 1
    grad = nu - area_vect 
    return grad

def grad_h_eps(v,X_source,X_target,nu,epsilon):
    n_target = len(X_target)
    expv = np.zeros(n_target)

    c = np.sum(np.abs(X_target-X_source)**2,axis=1)
    z = (v-c)/epsilon
    expv = nu * np.exp(z - np.max(z))  

    chi = expv/np.sum(expv)
    grad = nu - chi
    return grad

def h_eps(v,X_source,X_target,nu,epsilon):
    part1 = np.sum(v*nu)
    c = np.sum(np.abs(X_target-X_source)**2,axis=1)
    if epsilon == 0:
        part2 = np.min(c - v)
    else:
        z = (v-c)/epsilon
        expv = nu * np.exp(z - np.max(z))
        part2 = - epsilon * (np.log(np.sum(expv)) + np.max(z)) - epsilon
    h = part1 + part2
    return h

def W_sd(v,X_source,X_target,mu,nu,epsilon):
    n_source = len(X_source)
    h = np.zeros(n_source)
    for i in range(n_source):
        h[i] = h_eps(v,X_source[i],X_target,nu,epsilon)
    W = np.sum(h * mu)
    return W

