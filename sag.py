import numpy as np
import time
from scipy.stats import multivariate_normal

D = 3
p = 2

def sample_rho_batch(rho_list,nsamples):
    sample = np.zeros([nsamples,D])
    nrho = len(rho_list)
    for i in range(nsamples):
        rand = np.random.rand(1)
        idx = int(np.floor(nrho * rand))
        sample[i,:] = rho_list[idx].rvs()
        
    return sample

def gradient_SAG(v_eps,epsilon,n_target, n_source, X_source,X_target,nu,idx,p):
    expv = np.zeros(n_target)
    while np.sum(expv) == 0:
        z = np.max(v_eps-np.sum(abs(X_target-X_source[idx,:])**p,axis=1)/epsilon)
        expv = nu * np.exp(v_eps-np.sum(abs(X_target-X_source[idx,:])**p,axis=1)/epsilon - z)
        if np.sum(expv) == 0:
            print("simulate again")
    pi = expv/np.sum(expv)
    grad = - nu + pi
    return grad

def runSAG (epsilon,nb_iter,n_target,n_source,X_target,X_source,nu,alpha) :
    
    v_list = np.zeros([n_target,nb_iter])    

    # v_eps_bar = np.ones(n_target)
    v_eps = np.ones(n_target)

    grad_vect = np.zeros([n_target,n_source])
    grad_moy = np.zeros(n_target)

 
    for i in range(nb_iter):

        if i<n_source:
            n_grad = i+1
            idx = i
        else :
            n_grad = n_source
            idx = np.random.choice(range(n_source))

        v_list[:,i] = epsilon * v_eps
        grad_idx = gradient_SAG(v_eps,epsilon,n_target,n_source,X_source,X_target,nu,idx,p)
        grad_moy = grad_moy - grad_vect[:,idx]
        grad_vect[:,idx] = grad_idx
        grad_moy = grad_moy + grad_idx

        v_eps = v_eps - alpha/float(n_grad) * grad_moy

    v_SAG = epsilon * np.array(v_eps)

    return v_list

# %%
rho_list_source = []
nrho = 3
for i in range(nrho):
    mu1 = np.random.rand(D)
    sigma_tmp = np.random.rand(D,D)
    sigma1 = 0.01 *((sigma_tmp.T + sigma_tmp)+ D * np.eye(D))
    rho = multivariate_normal(mean = mu1,cov = sigma1)
    rho_list_source.append(rho)

rho_list_target = []
nrho = 3
for i in range(nrho):
    mu1 = np.random.rand(D)
    sigma_tmp = np.random.rand(D,D)
    sigma1 = 0.01 *((sigma_tmp.T + sigma_tmp)+ D * np.eye(D))
    rho = multivariate_normal(mean = mu1,cov = sigma1)
    rho_list_target.append(rho)


# %%
n_source = 15
n_target = 10

mu = np.ones(n_source)
mu = mu/np.sum(mu)

nu = np.ones(n_target)
nu = nu/np.sum(nu)

X_source = sample_rho_batch(rho_list_source,n_source) 
X_target = sample_rho_batch(rho_list_target,n_target)

n_it_SAG = 1000

# eps_list = [10**(-2)]
# n_eps = len(eps_list)
epsilon = 0.01

# n_alpha_SAG = 1
alpha = 0.003 / epsilon

# v_SAG = np.zeros([n_target,n_it_SAG ,n_eps,n_alpha_SAG])
v_SAG = np.zeros([n_target,n_it_SAG ,1,1])

# %%
# for i in range(n_eps):
#     epsilon = eps_list[i]
#     alpha_list_SAG = [0.003/epsilon]
#     n_alpha_SAG = len(alpha_list_SAG)
#     for j in range(n_alpha_SAG):
#         alpha = alpha_list_SAG[j]
#         t = time.time()
#         v_SAG[:,:,i,j] = runSAG(epsilon,n_it_SAG,n_target,n_source,X_target,X_source,nu,alpha)
#         tt = time.time() - t
#         print ("SAG, epsilon = "+str(epsilon)+', time elapsed : '+str(tt))

# %%
v_SAG[:,:,0,0] = runSAG(epsilon,n_it_SAG,n_target,n_source,X_target,X_source,nu,alpha)