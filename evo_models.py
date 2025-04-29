# This file will include all the WF models for the project

import numpy as np
import scipy.stats as stats

# Basic WF model: mutation, selection, drift
def WF(s,mu,N, G, seed = 0, p_init=0):
    # s: selection coefficient
    # mu: mutation rate
    # N: population size
    # G: number of generations
    # Returns the frequency of the mutant allele at generation G
    np.random.seed(seed)
    p = np.zeros((G+1,2)) # array to store the frequency of the mutant allele
    p[0] = np.array([1-p_init,p_init]) # initial frequency of the mutant allele
    w = 1+s # fitness of the mutant allele
    M = np.array([[1-mu, 0], 
                  [mu, 1]]) # mutation matrix
    S = np.array([[1, 0], 
                  [0, w]]) # selection matrix
    E = S@M # evolutionary matrix
    for i in range(G): # iterating over generations
        p[i+1] = E@p[i] # updating the frequency of the mutant allele
        p[i+1] /= np.sum(p[i+1]) # normalizing the frequency
        n = np.random.multinomial(N, p[i+1]) # sampling the number of mutants
        p[i+1] = n/N # updating the frequency of the mutant allele
    res = p[[10*(i+1) for i in range(G//10)], 1]
    return res

# # Wright-Fisher model with frequency-dependent selection
# def WF_FD(s, mu, N, G, seed = 0, p_init=0, alpha=8):
#     # s: selection coefficient
#     # mu: mutation rate
#     # N: population size
#     # G: number of generations
#     # Returns the frequency of the mutant allele at generation G
#     np.random.seed(seed)
#     p = np.zeros((G+1,2)) # array to store the frequency of the mutant allele
#     p[0] = np.array([1-p_init,p_init]) # initial frequency of the mutant allele
#     M = np.array([[1-mu, 0], [mu, 1]]) # mutation matrix

#     for i in range(G): # iterating over generations
#         if p[i,1] >= 0.05:
#             w = 1+alpha*s*p[i,1]*(1-p[i,1]) # fitness of the mutant allele  
#         else:
#             w = 1+s
#         S = np.array([[1, 0], [0, w]]) # selection matrix
#         E = S@M # evolutionary matrix
#         p[i+1] = E@p[i] # updating the frequency of the mutant allele
#         p[i+1] /= np.sum(p[i+1]) # normalizing the frequency
#         n = np.random.multinomial(N, p[i+1]) # sampling the number of mutants
#         p[i+1] = n/N # updating the frequency of the mutant allele
#     res = p[[10*(i+1) for i in range(G//10)], 1]
#     return res

# Wright-Fisher model with bottleneck events
def WF_bottleneck(s, mu, N, G, bottleneck, p_init=0, seed = 0):
    # s: selection coefficient
    # mu: mutation rate
    # N: population size
    # G: number of generations
    # bottleneck: dictionary of bottleneck times and their corresponding population sizes
    # Returns the frequency of the mutant allele at generation G
    np.random.seed(seed)
    max_N = N
    p = np.zeros((G+1,2)) # array to store the frequency of the mutant allele
    p[0] = np.array([1-p_init,p_init]) # initial frequency of the mutant allele
    M = np.array([[1-mu, 0], [mu, 1]]) # mutation matrix
    w = 1+s # fitness of the mutant allele
    S = np.array([[1, 0], [0, w]]) # selection matrix
    E = S@M # evolutionary matrix
    
    for i in range(G): # iterating over generations
        if i in bottleneck: # checking if there is a bottleneck event
            N = bottleneck[i] # updating the population size
        else:
            N = max_N
        p[i+1] = E@p[i] # updating the frequency of the mutant allele
        p[i+1] /= np.sum(p[i+1]) # normalizing the frequency
        n = np.random.multinomial(N, p[i+1]) # sampling the number of mutants
        p[i+1] = n/N # updating the frequency of the mutant allele
    res = p[[10*(i+1) for i in range(G//10)], 1]
    return res

# Wright-Fisher model with exponential DFE
def WF_DFE(mu,N, G, dist, p_init=0, seed = 0):
    # s: selection coefficient
    # mu: mutation rate
    # N: population size
    # G: number of generations
    # Returns the frequency of the mutant allele at generation G
    np.random.seed(seed)

    # DFE
    s_ = dist.rvs(G)

    p = np.zeros((G+1,2)) # array to store the frequency of the mutant allele
    p[0] = np.array([1-p_init,p_init]) # initial frequency of the mutant allele
    M = np.array([[1-mu, 0], [mu, 1]]) # mutation matrix
    for i in range(G): # iterating over generations
        w = 1+s_[i] # fitness of the mutant allele
        S = np.array([[1, 0], [0, w]]) # selection matrix
        E = S@M # evolutionary matrix
        p[i+1] = E@p[i] # updating the frequency of the mutant allele
        p[i+1] /= np.sum(p[i+1]) # normalizing the frequency
        n = np.random.multinomial(N, p[i+1]) # sampling the number of mutants
        p[i+1] = n/N # updating the frequency of the mutant allele
    res = p[[10*(i+1) for i in range(G//10)], 1]
    return res




# For 2-mutation misleading case-study
def combined_WF(s1, m1, s2, m2, s12, m12, m21, N, G, seed=0, noisy=False):
        
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert N > 0
    N = np.uint64(N)    
    
    # Order is: cnv, non-cnv
    
    w = np.array([1, 1 + s1, 1 + s2, 1 + s12], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m1 - m2, 0, 0, 0], # A0B0
                  [m1, 1-m12, 0, 0], # A1B0
                 [m2, 0, 1-m21, 0], # A0B1
                 [0, m12, m21, 1]], dtype='float64') # A1B1
    assert np.allclose(M.sum(axis=0), 1)
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(4)
    n[0] = 1 # wt at time 0
              
    # follow proportion of the population with mutations
    # here rows will be generation, columns (there is only one) is replicate population
    p12 = []
    # run simulation to generation G
    for t in range(G+1):    
        p = n/N  # counts to frequencies
        p12.append(p[-1])  # frequency of reported CNVs
        p = E @ p.reshape((4, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    
    ret = np.array(p12)[[10*(i+1) for i in range(G//10)]]
    if noisy:
        ret = ret + np.random.normal(loc=0, scale=0.02, size=(2*len(generation),))
    return ret


# For full 2-mutation task
def combined_WF_full(s1, m1, s2, m2, s12, m12, m21, N, G, seed=0, noisy=False):
        
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert N > 0
    N = np.uint64(N)    
    
    # Order is: cnv, non-cnv
    
    w = np.array([1, 1 + s1, 1 + s2, 1 + s12], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m1 - m2, 0, 0, 0], # A0B0
                  [m1, 1-m12, 0, 0], # A1B0
                 [m2, 0, 1-m21, 0], # A0B1
                 [0, m12, m21, 1]], dtype='float64') # A1B1
    assert np.allclose(M.sum(axis=0), 1)
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(4)
    n[0] = 1 # wt at time 0
              
    # follow proportion of the population with mutations
    # here rows will be generation, columns (there is only one) is replicate population
    p1 = []
    p2 = []
    # run simulation to generation G
    for t in range(G+1):    
        p = n/N  # counts to frequencies
        p1.append(p[1]+p[3])  # marginal frequency 1
        p2.append(p[2]+p[3])  # marginal frequency 2
        p = E @ p.reshape((4, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    
    ret1 = np.array(p1)[[10*(i+1) for i in range(G//10)]]
    ret2 = np.array(p2)[[10*(i+1) for i in range(G//10)]]
    ret = np.concatenate([ret1,ret2])
    if noisy:
        ret = ret + np.random.normal(loc=0, scale=0.02, size=(2*len(generation),))
    return ret
