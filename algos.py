from utilsOT import *

def runSGD (X_target,nu,rho_list_source,epsilon,nb_iter) :

    alpha = .8

    vlist = np.zeros([n_target,nb_iter])

    v = np.zeros(n_target)
    v_tilde = np.zeros(n_target)
    
    for i in range(nb_iter) :
        vlist[:,i] = v
        step = alpha*(1./np.sqrt(i+1))
        if epsilon == 0:
            n_samples = 1
            grad = grad_h_0(X_target,v_tilde,nu,rho_list_source,n_samples)
        else:
            grad = grad_h_eps(X_target,v_tilde,epsilon,nu,rho_list_source)
        v_tilde = v_tilde + step*grad
        v = 1./(i+1)*(v_tilde + i*v)
    
    return vlist

