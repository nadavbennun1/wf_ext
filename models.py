# This file will include all the WF models for the project

# Importing the required libraries
import numpy as np
import pandas as pd
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

# Wright-Fisher model with frequency-dependent selection
def WF_FD(s, mu, N, G, seed = 0, p_init=0, alpha=7):
    # s: selection coefficient
    # mu: mutation rate
    # N: population size
    # G: number of generations
    # Returns the frequency of the mutant allele at generation G
    np.random.seed(seed)
    p = np.zeros((G+1,2)) # array to store the frequency of the mutant allele
    p[0] = np.array([1-p_init,p_init]) # initial frequency of the mutant allele
    M = np.array([[1-mu, 0], [mu, 1]]) # mutation matrix

    for i in range(G): # iterating over generations
        w = 1+alpha*s*p[i,1]*(1-p[i,1]) # fitness of the mutant allele  
        S = np.array([[1, 0], [0, w]]) # selection matrix
        E = S@M # evolutionary matrix
        p[i+1] = E@p[i] # updating the frequency of the mutant allele
        p[i+1] /= np.sum(p[i+1]) # normalizing the frequency
        n = np.random.multinomial(N, p[i+1]) # sampling the number of mutants
        p[i+1] = n/N # updating the frequency of the mutant allele
    res = p[[10*(i+1) for i in range(G//10)], 1]
    return res

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
