import numpy as np
import pandas as pd
import torch
from sbi.utils import BoxUniform
from inference_utils import get_prior, get_dist
from models import WF, WF_FD, WF_bottleneck, WF_DFE

# Define the prior and constants
prior = get_prior()
N = int(1e8)
G = 200

# Define the simulator
def simulator(model, theta):
    s = 10**theta[0].item()
    mu = 10**theta[1].item()
    if model == 'WF':
        res = WF(s, mu, N, G)
    elif model == 'WF_FD':
        res = WF_FD(s, mu, N, G, p_init=0.05)
    elif model == 'WF_bottleneck':
        bottleneck = [int(0.001*N) for i in range(G//10)]
        res = WF_bottleneck(s, mu, N, G, bottleneck)
    elif model == 'WF_DFE':
        dist = get_dist(s)
        res = WF_DFE(mu, N, G, dist)
    return torch.tensor(res)

def export_sims(model, num_simulations):
    thetas_df = pd.DataFrame(np.zeros((num_simulations, 2)), columns=['s', 'mu'])
    x_df = pd.DataFrame(np.zeros((num_simulations, G//10)), columns=[f'x{10*i}' for i in range(G//10)])
    thetas = prior.sample((num_simulations,))
    for i in range(len(thetas)):
        theta = thetas[i]
        x_df.iloc[i,:] = np.array(simulator(model, theta))
        thetas_df.iloc[i,:] = np.array(theta)
    x_df.to_csv(f'{model}_sims.csv', index=False)
    thetas_df.to_csv(f'{model}_thetas.csv', index=False)

if __name__ == '__main__':
    # export_sims('WF', 1000)
    # export_sims('WF_FD', 1000)
    # export_sims('WF_bottleneck', 1000)
    export_sims('WF_DFE', 1000)