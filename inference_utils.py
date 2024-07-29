from sbi.utils import BoxUniform
import torch
from scipy import stats

# Define the prior
def get_prior():
    prior = BoxUniform(low=torch.tensor([-3, -10]), high=torch.tensor([0, -3]))
    return prior

def get_dist(s):
    lower, upper = 0.5*s, 1.5*s
    u, sigma = s, s/5
    dist = stats.truncnorm(
            (lower - u) / sigma, (upper - u) / sigma, loc=u, scale=sigma)
    return dist