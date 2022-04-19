#!/usr/bin/env python3
# Copyright 2020 ETH Zurich. All Rights Reserved.

import torch

class MLP(torch.nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 hl_dims: list):

        super().__init__()
        if len(hl_dims) < 1:
            raise ValueError(f"expected at least one hidden layer, got {len(hl_dims)}")

        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(input_dims, hl_dims[0], bias=True))
        self.layers.append(torch.nn.Tanh())

        for hl_dim0, hl_dim1 in zip(hl_dims[0:], hl_dims[:-1]):
            self.layers.append(torch.nn.Linear(hl_dim0, hl_dim1, bias=True))
            self.layers.append(torch.nn.Tanh())

        self.layers.append(torch.nn.Linear(hl_dims[-1], output_dims, bias=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def save_model_states(model, *,
                      xshift, xscale,
                      yshift, yscale,
                      path):
    """
    Save the model to a pickle file.
    """
    data = {"model": model,
            'xshift': xshift,
            'xscale': xscale,
            'yshift': yshift,
            'yscale': yscale}

    import pickle as pkl
    with open(path, "wb") as f:
        pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)


def load_model_states(path):
    """
    Load the model from a pickle file.
    """
    import pickle as pkl
    with open(path, "rb") as f:
        data = pkl.load(f)

    model = data['model']
    xshift = data['xshift']
    xscale = data['xscale']
    yshift = data['yshift']
    yscale = data['yscale']

    return model, xshift, xscale, yshift, yscale
