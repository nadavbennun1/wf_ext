import numpy as np

import pickle
import torch

from sbi.inference import NPSE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from inference_utils import get_prior

# Check prior, return PyTorch prior.
prior = get_prior()

theta = torch.load(f'test_sims/theta.pt')
x = torch.load(f'test_sims/x.pt')

inference = NPSE(prior, sde_type="ve")
_ = inference.append_simulations(theta, x).train()
posterior_npse = inference.build_posterior()




def sample_test(posterior, thetas, x, n_samples, model, arch):
    all_samples = torch.empty(n_samples, len(thetas), len(thetas[0]))
    all_samples[:,0] = posterior.set_default_x(x[0]).sample((n_samples,))
    for q in range(1,len(thetas)):
        all_samples[:,q] = posterior.set_default_x(x[q]).sample((n_samples,))
    torch.save(all_samples,f'test_sims/samples_{model}_{arch}.pt')
    return all_samples

models = ['WF', 'WF_DFE', 'WF_bottleneck']

for model in models:
    thetas = torch.load(f'test_sims/test_theta_{model}.pt')
    x = torch.load(f'test_sims/test_x_{model}.pt')
    sample_test(posterior_npse, thetas, x, 1000, model, 'npse')