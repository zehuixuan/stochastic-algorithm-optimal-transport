import numpy as np

from setting import *
from utilsOT import *

# def gradient(v_eps,epsilon):
#     expv = np.zeros(n_target)
#     while np.sum(expv) == 0:
#         Y = sample_rho(rho_list)
#         z = np.max((v_eps-np.sum((X-Y)**2,axis=1))/epsilon)
#         expv = nu * np.exp((v_eps-np.sum((X-Y)**2,axis=1))/epsilon - z)
#         #z = np.max(v_eps-np.sum((X-Y)**2,axis=1)/epsilon)
#         #expv = nu * np.exp(v_eps-np.sum((X-Y)**2,axis=1)/epsilon - z)
#         #if np.sum(expv) == 0:
#         #print "simulate again"
#     pi = expv/np.sum(expv)
#     grad = - nu + pi
#     return grad


def runSGD (epsilon,nb_iter) :
    grad_type = "one sample "

    alpha = .8
    #alpha = 1./epsilon
    n_eps = len(eps_list)


    vlist = np.zeros([n_target,nb_iter])

    v_eps_bar = np.ones(n_target)
    v_eps = np.ones(n_target)
    
    t = time.time()
    
    for i in range(nb_iter) :
        vlist[:,i] = v_eps_bar
        #vlist[:,i] = epsilon * v_eps_bar
        step = alpha*(1./np.sqrt(i+1))
        grad = gradient(v_eps,epsilon)
        v_eps = v_eps - step*grad
        v_eps_bar = 1./(i+1)*(v_eps + i*v_eps_bar)
        
    tt = time.time()-t
    
    print ("epsilon = "+str(epsilon)+', time elapsed : '+str(tt))
    return vlist


if __name__=="__main__":
    #########   SETTING    #########
    np.random.seed(3)
    
    epsilon = 0.01
    alpha = .5/epsilon
    n_target = 10 
    n_iter = 1000
    
    rho_list = generate_list_rho(n_rho=3)
    rho_list_target = generate_list_rho(n_rho=3)

    X_target = sample_rho_batch(rho_list_target,n_target)

    v_SGD = np.zeros([n_target,n_iter ,1,1])
    v_SGD[:,:,0,0] = runSGD(epsilon,n_iter,n_target,rho_list_source,X_target,nu,alpha)
