from utilsOT import *
from scipy.optimize import linprog

def runASGD_semidiscrete (nu,X_target,rho_list_source,epsilon,alpha,n_iter) :
    n_target = len(X_target)
    v_list = np.zeros([n_target,n_iter])

    v = np.zeros(n_target)
    v_tilde = np.zeros(n_target)
    
    for i in range(n_iter) :
        X_source = sample_rho(rho_list_source)
        v_list[:,i] = v
        step = alpha*(1./np.sqrt(i+1))
        if epsilon == 0:
            grad = grad_h_0(v_tilde, X_source, X_target,nu)
        else:
            grad = grad_h_eps(v_tilde, X_source,X_target,nu,epsilon)
        v_tilde = v_tilde + step*grad
        v = 1./(i+1)*(v_tilde + i*v)
    
    return v_list

def runASGD_discrete(nu,mu,X_target,X_source,epsilon,alpha,n_iter) :
    n_target = len(X_target)
    n_source = len(X_source)

    v_list = np.zeros([n_target,n_iter])

    v = np.zeros(n_target)
    v_tilde = np.zeros(n_target)
    
    for i in range(n_iter) :
        if i<n_source:
            idx = i
        else :
            idx = np.random.choice(range(n_source),p = mu)
        
        X_source_idx = X_source[idx,:]
        v_list[:,i] = v

        step = alpha*(1./np.sqrt(i+1))
        if epsilon == 0:
            grad = grad_h_0(v_tilde, X_source_idx, X_target,nu)
        else:
            grad = grad_h_eps(v_tilde, X_source_idx, X_target,nu,epsilon)
        v_tilde = v_tilde + step*grad
        v = 1./(i+1)*(v_tilde + i*v)
    
    return v_list

def runSAG (nu,mu,X_target,X_source,epsilon,alpha,n_iter) :
    n_target = len(X_target)
    n_source = len(X_source)

    v_list = np.zeros([n_target,n_iter])
    
    v = np.zeros(n_target)
    grad_vect = np.zeros([n_target,n_source])
    grad_moy = np.zeros(n_target)
    sum_mu = 0
 
    for i in range(n_iter):

        if i<n_source:
            idx = i
            sum_mu += mu[idx]
        else :
            idx = np.random.choice(range(n_source))
            sum_mu = 1
        
        v_list[:,i] = v
        
        grad_moy = grad_moy - grad_vect[:,idx]
        X_source_idx = X_source[idx,:]
        if epsilon == 0:  
            grad_idx = mu[idx] * grad_h_0(v,X_source_idx,X_target,nu)
        else:
            grad_idx = mu[idx] * grad_h_eps(v,X_source_idx,X_target,nu,epsilon)
        grad_vect[:,idx] = grad_idx
        grad_moy = grad_moy + grad_idx
        v = v + alpha / sum_mu * grad_moy

    return v_list

def runLP(nu,mu,X_target,X_source):
    n_source = len(X_source)
    n_target = len(X_target)
    
    C=np.zeros((n_source,n_target))
    for i in range(n_source):
        for j in range(n_target):
            C[i,j] = np.sum(np.abs(X_source[i]-X_target[j])**2)
    C = C.reshape(n_source*n_target)

    Id = np.identity(n_target)
    A_eq_1 = np.concatenate((Id,Id),axis=1)
    for k in range(n_source-2):
        A_eq_1 = np.concatenate((A_eq_1,Id),axis=1)

    A_eq_2 = np.zeros((n_source,n_source*n_target))
    for k in range(n_source):
        A_eq_2[k,k*n_target:(k+1)*n_target] = np.ones((1,n_target))

    A_eq = np.concatenate((A_eq_1,A_eq_2), axis=0)

    b_eq_1 = nu
    b_eq_2 = mu
    b_eq = np.concatenate((b_eq_1,b_eq_2),axis=0)

    res = linprog(c=C, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    P = res.x
    W = np.sum(C*P)
    return P.reshape(n_source,n_target), W


def calculate_W_list_SAG (nu,mu,X_target,X_source,epsilon,alpha,n_iter) :
    n_target = len(X_target)
    n_source = len(X_source)
    
    W_list = np.zeros([n_iter//100])
    
    v = np.zeros(n_target)
    grad_vect = np.zeros([n_target,n_source])
    grad_moy = np.zeros(n_target)
    sum_mu = 0
 
    for i in range(n_iter):

        if i<n_source:
            idx = i
            sum_mu += mu[idx]
        else :
            idx = np.random.choice(range(n_source))
            sum_mu = 1
        
        if (i+1) % 100 == 0:
            W_list[((i+1)//100)-1] = W_sd(v,X_source,X_target,mu,nu,epsilon)

        grad_moy = grad_moy - grad_vect[:,idx]
        X_source_idx = X_source[idx,:]
        if epsilon == 0:  
            grad_idx = mu[idx] * grad_h_0(v,X_source_idx,X_target,nu)
        else:
            grad_idx = mu[idx] * grad_h_eps(v,X_source_idx,X_target,nu,epsilon)
        grad_vect[:,idx] = grad_idx
        grad_moy = grad_moy + grad_idx
        v = v + alpha / sum_mu * grad_moy

    return W_list
