# Test NPE, NLE, NRE

import torch
import pickle
import argparse

#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model") # model name
parser.add_argument('-a', "--architecture") # SBI architecture
parser.add_argument('-s', "--samples") # number of samples from the posterior
parser.add_argument('-ss', "--save_samples", action='store_true') # whether to save samples
parser.add_argument('-rej', "--rej", action='store_true') # whether to use rejection posterior
args = parser.parse_args()


# Parsed args
model = str(args.model) # WF/WF_DFE/WF_bottleneck
arch = args.architecture # npe/nle/nre
samples = int(args.samples) # no. samples per observation
ss = args.save_samples # ss -> save samples

# Load the posterior with pickle
posterior = pickle.load(open(f'posteriors/posterior_{arch}.pkl', 'rb'))

# Load the test thetas
thetas = torch.load(f'test_sims/test_theta_{model}.pt')
x = torch.load(f'test_sims/test_x_{model}.pt')


# Export samples of NDEs per test observation
# Use batch sampling with jumps of 100 to optimize computation time
def sample_test(posterior, thetas, x, n_samples, by=100):
    all_samples = posterior.sample_batched((n_samples,), x[:by])
    k = len(x)//by
    for i in range(1,k):
        samps = posterior.sample_batched((n_samples,), x[by*i:by*(i+1)])
        all_samples = torch.cat([all_samples,samps], axis=1)
    if ss: 
        torch.save(all_samples, f'test_sims/samples_{model}_{arch}.pt')
    return all_samples

sample_test(posterior, thetas, x, samples)