# Adapted from Hugging Face implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

def select_neurons(neuron_stat, method, k):
    if method == 'topk':
        weight, indices = torch.topk(neuron_stat, k, dim=-1)
    else:
        raise NotImplementedError

    return weight, indices

def get_llama_griffin(model, topk_ratio, mode):
    config = model.config
    for i, l in enumerate(model.model.layers):
        new_mlp = LlamaMLP(config, topk_ratio, mode)

        new_mlp.gate_proj = l.mlp.gate_proj
        new_mlp.up_proj = l.mlp.up_proj
        new_mlp.down_proj = l.mlp.down_proj
        new_mlp.act_fn = l.mlp.act_fn

        l.mlp = new_mlp
    
    return model


class LlamaMLP(nn.Module):
    def __init__(self, config, k_factor, mode):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu
        
        self.k_factor = k_factor
        self.mode = mode
        assert self.mode in ['gen', 'class']


    def prepare_reduced_weights(self, topk_indices):
        assert topk_indices.shape[0] == 1 # Batch size 1
        
        self.gate_proj_reduced = nn.Linear(self.gate_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.up_proj_reduced = nn.Linear(self.up_proj.weight.data.shape[1], len(topk_indices), bias=False)
        self.down_proj_reduced = nn.Linear(len(topk_indices), self.down_proj.weight.data.shape[0], bias=False)
        topk_indices = topk_indices[0]

        self.gate_proj_reduced.weight.data = self.gate_proj.weight.data[topk_indices]
        self.up_proj_reduced.weight.data = self.up_proj.weight.data[topk_indices]
        self.down_proj_reduced.weight.data = self.down_proj.weight.data[:, topk_indices]
    

    def forward(self, x):

        k_factor = self.k_factor
        if self.mode == 'gen':
            if x.shape[1] > 1:
                int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

                # GRIFFIN Expert Selection
                k = int(int_states.shape[-1] * k_factor)
                neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1))).norm(dim=1) # B, D
                topk_weight, topk_indices = select_neurons(neuron_stat, 'topk', k)
                self.prepare_reduced_weights(topk_indices)
                        
                down_proj = self.down_proj(int_states)

            else:
                if k_factor == 0.0:
                    down_proj = 0 * x 
                else:
                    down_proj =self.down_proj_reduced(self.act_fn(self.gate_proj_reduced(x)) * self.up_proj_reduced(x))
        else:
            raise NotImplementedError
        return down_proj