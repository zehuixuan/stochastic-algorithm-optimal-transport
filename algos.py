from utilsOT import *

def runASGD (nu,X_target,rho_list_source,epsilon,alpha,n_iter) :

    v_list = np.zeros([n_target,n_iter])

    v = np.zeros(n_target)
    v_tilde = np.zeros(n_target)
    
    for i in range(n_iter) :
        v_list[:,i] = v
        step = alpha*(1./np.sqrt(i+1))
        if epsilon == 0:
            n_samples = 1
            grad = grad_h_0(v_tilde, rho_list_source, X_target,nu,n_samples)
        else:
            grad = grad_h_eps(v_tilde, rho_list_source,X_target,nu,epsilon)
        v_tilde = v_tilde + step*grad
        v = 1./(i+1)*(v_tilde + i*v)
    
    return v_list

def runSAG (nu,mu,X_target,X_source,epsilon,alpha,n_iter) :
    
    v_list = np.zeros([n_target,n_iter])
    
    v = np.zeros(n_target)
    grad_vect = np.zeros([n_target,n_source])
    grad_moy = np.zeros(n_target)
 
    for i in range(n_iter):

        if i<n_source:
            n_grad = i+1
            idx = i
        else :
            n_grad = n_source
            idx = np.random.choice(range(n_source))

        # v_list[:,i] = epsilon * v
        v_list[:,i] = v
        
        grad_moy = grad_moy - grad_vect[:,idx]
        grad_idx = grad_SAG(v,X_source,idx,X_target,nu,epsilon)
        grad_vect[:,idx] = grad_idx
        grad_moy = grad_moy + grad_idx
        v = v + alpha/n_grad * grad_moy

    return v_list
