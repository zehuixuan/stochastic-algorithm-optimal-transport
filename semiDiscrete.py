import numpy as np
import matplotlib.pyplot as plt


from utilsOT import *
from algos import *


if __name__=="__main__":
    #########   SETTING    #########
    np.random.seed(3)
    
    n_iter = 100000
    n_target = 10

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
    v_ASGD[:,:,0,0] = runASGD_semidiscrete(nu,X_target,rho_list_source, epsilon=0, alpha = 0.8, n_iter=n_iter)

    #########   Averaged SGD Regularized, epsilon = 0.01   #########
    v_ASGDr = np.zeros([n_target,n_iter,1,1])
    v_ASGDr[:,:,0,0] = runASGD_semidiscrete(nu,X_target,rho_list_source, epsilon=0.01, alpha = 0.8, n_iter=n_iter)

    #########  Discretization of mu  #########
    # sample from continuous measure mu
    n_source = 1000
    X_source = sample_rho_batch(rho_list_source,n_source)
    mu = np.ones(n_source)
    mu = mu/np.sum(mu)

    #########   Averaged SGD Non-regularizied, Discretization   #########
    v_ASGDd = np.zeros([n_target,n_iter ,1,1])
    v_ASGDd[:,:,0,0] = runASGD_discrete(nu,mu,X_target,X_source, epsilon=0, alpha = 0.8, n_iter=n_iter)

    #########   Averaged SGD Regularized, Discretization, epsilon = 0.01   #########
    v_ASGDrd = np.zeros([n_target,n_iter ,1,1])
    v_ASGDrd[:,:,0,0] = runASGD_discrete(nu,mu,X_target,X_source, epsilon=0.01, alpha = 0.8, n_iter=n_iter)

    #########   SAG Non-regularized, Discretization   #########
    v_SAG = np.zeros([n_target,100000 ,1,1])
    v_SAG[:,:,0,0] = runSAG(nu,mu,X_target,X_source,epsilon = 0,alpha=0.001,n_iter=100000)


    #########   SAG Regularized, Discretization, epsilon = 0.01   #########

    # epsilon = 0.0001
    # alpha = 0.001
    # n_iter = 100000
    v_SAGr = np.zeros([n_target,100000 ,1,1])
    v_SAGr[:,:,0,0] = runSAG(nu,mu,X_target,X_source,epsilon = 0.01,alpha=0.001,n_iter=100000)


    #########   Plot   #########

    for idx in range(n_target):
        plt.figure()
        plt.plot(v_ASGD[idx,:,0,0],label = 'ASGD')
        plt.plot(v_SAGr[idx,:,0,0], label = 'SAGr')
        plt.plot(v_SAG[idx,:,0,0], label = 'SAG')
        plt.plot(v_ASGDd[idx,:,0,0], label = 'ASGDr')
        plt.plot(v_ASGDr[idx,:,0,0], label = 'ASGDr')
        plt.plot(v_ASGDrd[idx,:,0,0], label = 'ASGDrd')
        plt.xscale('log')
        plt.legend()
        plt.show()