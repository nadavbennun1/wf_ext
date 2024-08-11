# inference with NPE
from models import WF
from sbi.inference.base import infer
from inference_utils import get_prior
import torch
import pickle

# Define the prior
prior = get_prior()

# Define the simulator
def simulator(theta):
    s = 10**theta[0].item()
    mu = 10**theta[1].item()
    N = int(1e8)
    G = 200
    res = WF(s, mu, N, G)
    return torch.tensor(res)

# inference
posterior = infer(simulator, prior, method='SNRE_A', num_simulations=10000)

# Save the posterior with pickle
with open('posterior_bnre.pkl', 'wb') as f:
    pickle.dump(posterior, f)