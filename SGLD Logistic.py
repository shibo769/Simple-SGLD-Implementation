'''
Load Packages
'''

from scipy import stats
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad
from sklearn import preprocessing
import time
from sklearn.preprocessing import StandardScaler

def sigmoid(theta, X):
    """
    theta : param
    X     : data matrix
    """
    return 1 / (1 + np.exp(-np.matmul(X, theta)))

def neg_log_loss(theta,X,Y):
    """
    loss function
    """
    N,d = X.shape
    log_prior =  -0.5 * np.dot(theta.T, theta) - 1 * 0.5 * d * np.log(2 * np.pi)
    func = (-1) * (log_prior + log_lik(theta, X, Y))
    return func

def batch(X, Y, N_obs, batch_size, replace = True):
    """
    Get random batch via randon choice sampleing function
    """
    batch_ind = np.random.choice(N_obs, size=batch_size, replace = replace)
    return X[batch_ind, :], Y[batch_ind]

def step_size(X):
    """
    Get the step size from the paper
    gamma = (1 - delta / 4)^(-1)
    where delta is the max{lambda of X'X}
    """
    delta = np.max(scipy.linalg.eigh(np.dot(X.T, X), eigvals_only = True))
    return 1 / (1 + 0.025 * delta)

def SGLD(n_steps, burn_in, batch_size, theta_init, gamma, X, Y, thin=1):
    """
    n_step: # of iterations from the paper 1 / gamma
    burn-in : first 10% burn-in
    batch_size: batch_size
    theta_init: initial theta for the algo
    gamma: step size
    X : degisn matrix
    Y : response variable
    """
    #LMC iterate over whole data set
    #theta_init must be an array
    theta = theta_init.copy()
    thetas = []
    N,_ = X.shape
    for i in range(n_steps):
        #print("current theta is", theta)
        X_batch,Y_batch = batch(X, Y, N, batch_size)
        proposal_loc = 0.5*gamma*(grad_prior(theta) + (N/batch_size) * \
                                  grad_lik(theta, X_batch, Y_batch))
        # This is where the difference between SGLD and SGD is 
        theta = theta + stats.multivariate_normal.rvs(mean=proposal_loc, cov=gamma) 
        if  i >= burn_in and i % thin == 0:
            thetas.append(theta.copy())
            
    return np.array(thetas)

def SGD(n_steps, burn_in, batch_size, theta_init, gamma, X, Y, thin=1):
    #theta_init must be an array
    theta = theta_init.copy()
    thetas = []
    N,_ = X.shape
    for i in range(n_steps):
        #print("current theta is", theta)
        X_batch,Y_batch = batch(X, Y, N, batch_size)
        proposal_loc = 0.5*gamma*(grad_prior(theta) + (N/batch_size) * \
                                  grad_lik(theta, X_batch, Y_batch))
        theta = theta + proposal_loc
        if  i >= burn_in and i % thin == 0:
            thetas.append(theta.copy())
    return np.array(thetas)

# Log likelihood
log_lik = lambda theta ,X ,Y: np.sum((-1 * Y * np.log(1 + np.exp(-1 * np.matmul(X, theta)))\
                                    + (Y - 1) * np.log(1 + np.exp(np.matmul(X, theta)))))
### Bayesian Logistic Model
grad_lik   = grad(log_lik) # get gradient of log likelihood
grad_prior = lambda theta: -1*theta # set simple prior

if __name__ == "__main__":
    ### Data Simulation
    theta      = np.array([-2.0, 2.0, -1.0]) # theta
    d          = theta.shape[0] # dimension
    true_mean  = [0.0, 0.0] # true mean
    true_cov   = 2 * np.identity(2) # true covariance
    N_obs      = 10**5 # # of obs
    batch_size = 1000
    seed       = 99 
    
    # Data Matrix
    X0         = stats.multivariate_normal.rvs(mean = true_mean,
                                              cov = true_cov,
                                              size = N_obs,
                                              random_state = seed)
    intercept  = np.ones((N_obs,1), dtype=float)
    X          = np.concatenate((X0, intercept), axis = -1)
    theta_init = np.ones(d) # initial theta
    
    ### Data under Logistic Model
    pi         = sigmoid(theta, X)
    Y          = stats.bernoulli.rvs(pi)
    
    ### Optimized Theta
    theta_ests = scipy.optimize.minimize(neg_log_loss, x0 = theta_init,args = (X,Y),method = 'BFGS')
    theta_opt = theta_ests['x']
    print("The optimized thetas are:",theta_opt)
    
    # SGLD and SGD
    gamma      = step_size(X) # largest eigvalue of X'X from paper
    n_steps    = int(1/(gamma)) # 1/gamma iterations from paper
    burn_in    = int(n_steps*0.1) # 10% burn-in from paper
    
    thetas_SGLD = SGLD(n_steps = n_steps,burn_in = burn_in, batch_size = 1000,
                       theta_init = theta_init, gamma = gamma, X = X, Y = Y)
    thetas_SGD  = SGD(n_steps = n_steps,burn_in = burn_in, batch_size = 1000,
                     theta_init = theta_init, gamma = gamma, X = X, Y = Y)
    
    SGLD_theta_bar = np.mean(thetas_SGLD, axis = 0)
    SGD_theta_bar  = np.mean(thetas_SGD, axis = 0)
    
    # plots
    plt.figure()
    plt.plot(thetas_SGLD[:, 0], label = "SGLD trace")
    plt.plot(thetas_SGD[:, 0], "orange", alpha = 0.6, label = "SGD trace")
    plt.legend()
    plt.show()
    
    #######################################
    theta_l2_diff = {"SGLD":[],"SGD":[]}
    gamma0 = step_size(X)
    inds = [   100,    200,    300,    400,    500,    600,    700,    800,    900,
   1000,   2000,   3000,   4000,   5000,   6000,   7000,   8000,   9000,
  10000,  20000,  30000,  40000,  50000,  60000,  70000,  80000,  90000,
 100000]
    gamma_list = []
    for i,ind in enumerate(inds):
        #####Data Truncation#####
        X_t = X[:int(ind),:]
        Y_t = Y[:int(ind)]
        print("*******At the iteration ",i,"*******")
        print("data dim ", X_t.shape)
        N = X_t.shape[0]
        print(ind)
        
        #####Weights Optimization#####
        theta_ests = scipy.optimize.minimize(neg_log_loss, x0 = [1.0,1.0,1.0],args = (X_t,Y_t),method = 'BFGS')
        theta_opt = theta_ests['x']
        
        #####Parameters Set-up######
        gamma0 = step_size(X_t)
        gamma_list.append(gamma0)
        n_steps = int(1/(gamma0))
        print("step size is", gamma0)
        print("num of mcmc runs is ", n_steps)
        burn_in = int(n_steps*0.1)
        print("burn-in period is ", burn_in)
        batch_size = int(np.sqrt(N))
        print("batch size is ", batch_size)
        
        #####Implementation#####
        kwargs = dict(n_steps = n_steps, burn_in = burn_in, theta_init = theta_init,
                      gamma = gamma, X = X_t, Y = Y_t)
        thetas_SGLD = SGLD(batch_size = batch_size,**kwargs)
        thetas_SGD = SGD(batch_size = batch_size,**kwargs)
        
        print("The opt weights are ", theta_opt)
        
        thetas_SGLD_mean = np.mean(thetas_SGLD, axis = 0)
        diff_SGLD = np.mean(thetas_SGLD, axis = 0) - theta_opt
        print("The SGLD mean is ",thetas_SGLD_mean)
        
        thetas_SGD_mean = np.mean(thetas_SGD, axis = 0)
        diff_SGD = thetas_SGD_mean - theta_opt
        print("The SGD mean is ",thetas_SGD_mean)

        theta_l2_diff["SGLD"].append(np.linalg.norm(diff_SGLD))
        theta_l2_diff["SGD"].append(np.linalg.norm(diff_SGD))

        print("\n")
        
    ### Distance plot
    plt.figure()
    plt.plot(inds,theta_l2_diff["SGLD"],label = "SGLD")
    plt.plot(inds,theta_l2_diff["SGD"],label = "SGD")
    plt.legend()
    plt.show()
    
    # gamma plot converes to 0
    plt.plot(inds, gamma_list)
    
    