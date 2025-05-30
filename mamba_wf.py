# Code by azhar labs, 2023, "Building Mamba from Scratch: A Comprehensive Code Walkthrough", Medium.
# I didn't use different H states for the recurrent update mechanism

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from einops import rearrange
from tqdm import tqdm

import math
import os
torch.autograd.set_detect_anomaly(True)

# Configuration flags and hyperparameters
USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0


class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, batch_size, device):
        # d_model: model dimension (D/channels in paper)
        # seq_len: data length (20/40)
        super(S6, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.batch_size = batch_size
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(self.batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(self.batch_size, self.seq_len, self.state_size, device=device)

        self.delta = torch.zeros(self.batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(self.batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(self.batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # h should have dimensions [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(self.batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(self.batch_size, self.seq_len, self.d_model, device=device)

    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))

        return self.dA, self.dB

    def forward(self, x):
        # Refer to Algorithm 2 in the MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        # Uses ZOH as in MAMBA
        self.discretization()

        h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
        y = torch.zeros_like(x)

        h =  torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

        # y needs to have a shape of [batch_size, seq_len, d_model]
        y = torch.einsum('bln,bldn->bld', self.C, h)

        return y



class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, batch_size, device):
        super(MambaBlock, self).__init__()

        self.inp_proj = nn.Linear(d_model, 2*d_model, device=device)
        self.out_proj = nn.Linear(2*d_model, d_model, device=device)

        # For residual skip connection
        self.D = nn.Linear(d_model, 2*d_model, device=device)

        # Batch size
        self.batch_size = batch_size
        
        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True

        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(seq_len, 2*d_model, state_size, batch_size, device)

        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)

        # Add linear layer for conv output
        self.conv_linear = nn.Linear(2*d_model, 2*d_model, device=device)

        # rmsnorm
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        """
        x_proj.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv.shape = torch.Size([batch_size, seq_len, 2*d_model])
        x_conv_act.shape = torch.Size([batch_size, seq_len, 2*d_model])
        """
        # Refer to Figure 3 in the MAMBA paper

        x = self.norm(x)

        x_proj = self.inp_proj(x)

        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)

        x_conv_act = F.silu(x_conv)

        # Add linear layer for conv output
        x_conv_out = self.conv_linear(x_conv_act)

        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual

        x_out = self.out_proj(x_combined)

        return x_out

    
class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, batch_size, device):
        super(Mamba, self).__init__()
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size, batch_size, device)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size, batch_size, device)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size, batch_size, device)
        
    def forward(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str ='cuda:1'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output