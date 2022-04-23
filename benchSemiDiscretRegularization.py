
############## ############## ############## ############## ############## 
########## Unregularized vs Regularized for various epsilon ############## 
############## ############## ############## ############## ############## 
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from random import randint
import sklearn as sk
from sklearn.neighbors import NearestNeighbors
from scipy.signal import convolve2d
import sys

from setting import *
from utilsOT import *
from algos import *

# ## Standard OT



def runBench(n_target, i_run, n_iter_comparaison, n_iter_SGD_opt):
    vlist = runSGD(X_target,nu,rho_list_source,epsilon,n_iter_SGD_opt)
    v_opt = vlist[:,-1]

    n_it_SGD = n_iter_comparaison

    v_SGD = np.zeros([n_target,n_it_SGD,n_eps])

    for i in range(n_eps):
        epsilon = eps_list[i]
        v_SGD[:,:,i] = runSGD(X_target,nu,rho_list_source,epsilon,n_it_SGD)


    kmax = np.shape(v_SGD)[1]
    lmax = np.shape(vlist)[1]

    n_size_err = min(kmax,lmax)


    err_SGD = np.zeros([n_size_err,n_eps])
    err_v = np.zeros(n_size_err)

    for l in range(n_size_err):
        err_v[l] = np.linalg.norm(vlist[:,l] - np.mean(vlist[:,l]) - v_opt + np.mean(v_opt))/np.linalg.norm(v_opt - np.mean(v_opt))

    for i in range(n_eps):
        epsilon = eps_list[i]
        for k in range(n_size_err):
            err_SGD[k,i] = np.linalg.norm(v_SGD[:,k,i]- np.mean(v_SGD[:,k,i]) - v_opt + np.mean(v_opt))/np.linalg.norm(v_opt-np.mean(v_opt))


    filename_reg = "/home/marco/temp/numpy_arrays/SemiDiscretRegularization/err_SGD_unreg_"+str(i_run)+'_batch_'+str(arg)
    np.save(filename_reg,err_v)

   
    filename = "/home/marco/temp/numpy_arrays/SemiDiscretRegularization/err_SGD_all_eps_"+str(i_run)+'_batch_'+str(arg)
    np.save(filename,err_SGD)


if __name__=="__main__":
    #########   SETTING    #########

    # continuous measure
    np.random.seed(3)

    rho_list_source = generate_list_rho(3) 
    rho_list_target = generate_list_rho(3)

    # discrete measure
    n_target = 10 
    X_target = sample_rho_batch(rho_list_target,n_target)

    nu = np.random.rand(n_target)
    nu = nu/np.sum(nu) 

    eps_list = [10**(-1),10**(-2),10**(-3),10**(-4)]
    #eps_list = [10**(-1),10**(-2)]

    n_eps = len(eps_list)




    n_iter_SGD_opt = 10**7
    n_iter_comparaison = 5*10**5

    nruns = 5
    arg = sys.argv[1]


    for i_run in range(nruns):
    
        print ("-------------    "+str(i_run)+"   ----------")

        # X_target = sample_rho_batch(rho_list_target,n_target)

    
        runBench(n_target, i_run, n_iter_comparaison, n_iter_SGD_opt)









