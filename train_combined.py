# Training of three NDEs: NPE, NLE, NRE
from evo_models import combined_WF_full
from inference_utils import get_prior_combined
from sbi.inference import NPE, NLE, NRE
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

import numpy as np
import torch
import pickle

# Algorithms
alg_list = [NPE, NLE, NRE]
alg_strs = ['npe', 'nle', 'nre']


# Define the prior
prior = get_prior_combined()

# Training data
theta = torch.load('test_sims/theta_combined.pt')
x = torch.load('test_sims/x_combined.pt')

for i in range(len(alg_list)):
    alg = alg_list[i]
    str_alg = alg_strs[i]
    inference = alg(prior).append_simulations(theta, x)
    density_estimator = inference.train(max_num_epochs=100)
    posterior = inference.build_posterior(density_estimator)
    # Save the posterior with pickle
    with open(f'posteriors/posterior_{str_alg}_combined.pkl', 'wb') as f:
        pickle.dump(posterior, f)