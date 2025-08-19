import torch.nn as nn
import torch.nn.init as init
import torch


def custom_weight_initialization(module):
    device = torch.device("mps")
    gen = torch.Generator(device=device)  
    gen.manual_seed(1000) 

    if isinstance(module, nn.Linear):
        # Xavier initialization for linear layers
        init.xavier_uniform_(module.weight, generator=gen)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embedding layers
        init.normal_(module.weight, mean=0, std=0.01, generator=gen)
    elif isinstance(module, nn.LayerNorm):
        # Ones for gamma and zeros for beta in LayerNorm
        init.ones_(module.weight)
        init.zeros_(module.bias)