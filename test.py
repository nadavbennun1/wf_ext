# inference with NPE
from evo_models import WF, WF_bottleneck, WF_DFE
import torch
import pickle
# import time
import argparse
from sbi.inference.posteriors import RejectionPosterior

#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model") # model name
parser.add_argument('-a', "--architecture") # SBI architecture
parser.add_argument('-s', "--samples") # number of samples from the posterior
parser.add_argument('-ss', "--save_samples", action='store_true') # whether to save samples
parser.add_argument('-rej', "--rej", action='store_true') # whether to use rejection posterior
args = parser.parse_args()


# Define the prior and simulator
model = str(args.model)
arch = args.architecture
samples = int(args.samples)
ss = args.save_samples
rej = args.rej

# Load the posterior with pickle
posterior = pickle.load(open(f'posteriors/posterior_{arch}.pkl', 'rb'))

# Load the test thetas
thetas = torch.load(f'test_sims/test_theta_{model}.pt')
x = torch.load(f'test_sims/test_x_{model}.pt')



def sample_test(posterior, thetas, x, n_samples, by=100, rej=False):
    if rej:
        posterior = RejectionPosterior(posterior.potential_fn, prior)
    all_samples = posterior.sample_batched((n_samples,), x[:by])
    k = len(x)//by
    for i in range(1,k):
        samps = posterior.sample_batched((n_samples,), x[by*i:by*(i+1)])
        all_samples = torch.cat([all_samples,samps], axis=1)
    if ss:
        torch.save(all_samples, f'test_sims/samples_{model}_{arch}.pt')
    return all_samples

sample_test(posterior, thetas, x, samples)