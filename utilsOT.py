from setting import *
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors

####### gaussian mixture & samples #######

def generate_list_rho(n_rho = 3):
    rho_list = []
    for i in range(n_rho):
        mu1 = np.random.rand(D)
        sigma_tmp = np.random.rand(D,D)
        sigma1 = 0.01 *((sigma_tmp.T + sigma_tmp)+ D * np.eye(D))
        rho = multivariate_normal(mean = mu1,cov = sigma1)
        rho_list.append(rho)
    return rho_list

def sample_rho_batch(rho_list,nsamples):
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

def grad_h_eps(X_target,v,epsilon,nu,rho_list_source):
    expv = np.zeros(n_target)
    while np.sum(expv) == 0:
        X_source = sample_rho(rho_list_source)
        c = np.sum((X_target-X_source)**2,axis=1)
        z = (v-c)/epsilon
        expv = nu * np.exp(z - np.max(z))  
        print(expv)     
    chi = expv/np.sum(expv)
    grad = nu - chi
    print(grad)
    return grad

def grad_h_0(X_target,v,nu,rho_list_source,n_samples):
    Xv = np.c_[X_target,np.sqrt(-v-np.min(-v))]
    kdTree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Xv)
    Y = sample_rho(rho_list_source)
    if n_samples == 1:
        Yv = np.reshape(np.hstack([Y,0]),[1,-1])
    else:
        Yv = np.c_[Y,np.zeros(n_samples)]
    neighbors = kdTree.kneighbors(Yv,n_neighbors=1,return_distance=False)
    area_vect = np.zeros(n_target)
    for i in range(n_samples):
          area_vect[neighbors[i]] += 1
    area_vect = area_vect/n_samples  
    grad = nu - area_vect 
    return grad



