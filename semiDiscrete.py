import numpy as np
import matplotlib.pyplot as plt

from setting import *
from utilsOT import *
from algos import *


if __name__=="__main__":
    #########   SETTING    #########
    np.random.seed(3)
    
    n_iter = 100000
    
    rho_list_source = generate_list_rho(3) 
    rho_list_target = generate_list_rho(3)

    X_target = sample_rho_batch(rho_list_target,n_target)
    nu = np.ones(n_target)
    nu = nu/np.sum(nu) 

    #########   Averaged SGD Non-regularizied   #########
    # epsilon = 0
    # alpha = 0.8
    # n_iter = 10000
    v_ASGD = np.zeros([n_target,n_iter ,1,1])
    v_ASGD[:,:,0,0] = runASGD(nu,X_target,rho_list_source, epsilon=0, alpha = 0.8, n_iter=n_iter)

    #########   SAG Regularized, Discretization of mu ,epsilon = 0.01   #########
    # sample from continuous measure mu
    X_source = sample_rho_batch(rho_list_source,n_source)
    mu = np.ones(n_source)
    mu = mu/np.sum(mu)

    # epsilon = 0.0001
    # alpha = 0.001
    # n_iter = 100000
    v_SAG = np.zeros([n_target,100000 ,1,1])
    v_SAG[:,:,0,0] = runSAG(nu,mu,X_target,X_source,epsilon = 0.0001,alpha=0.001,n_iter=100000)


