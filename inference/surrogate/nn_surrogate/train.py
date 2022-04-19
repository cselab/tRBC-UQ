#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import pathlib
import sys
from torch.utils.data import TensorDataset, DataLoader
import torch

from utils import TemporaryNumpySeed
from learning import (MLP,
                      init_weights,
                      train_model,
                      save_model_states)

def random_split(x: np.ndarray,
                 split_sizes: list,
                 seed: int):
    n = len(x)
    assert np.sum(split_sizes) == n

    order = np.arange(n)

    with TemporaryNumpySeed(seed):
        np.random.shuffle(order)

    x = x[order,:]

    split_data = list()
    start = 0
    for s in split_sizes:
        split_data.append(x[start:start+s,:])
        start += s
    return split_data


def train(df: pd.DataFrame,
          input_var: list,
          output_var: list,
          output_path: str,
          hidden_layers: list,
          train_ratio: float=0.8,
          batch_size: int=128,
          plot_errors: bool=False):

    nsamples, _ = df.shape

    x = torch.zeros((nsamples, len(input_var )))
    y = torch.zeros((nsamples, len(output_var)))

    xshift = []
    xscale = []

    for i, name in enumerate(input_var):
        data = df[name].to_numpy()
        shift = np.mean(data)
        scale = np.std(data)
        data = (data - shift) / scale
        x[:,i] = torch.from_numpy(data)
        xshift.append(shift)
        xscale.append(scale)

    yshift = []
    yscale = []

    for i, name in enumerate(output_var):
        data = df[name].to_numpy()
        shift = np.mean(data)
        scale = np.std(data)
        data = (data - shift) / scale
        y[:,i] = torch.from_numpy(data)
        yshift.append(shift)
        yscale.append(scale)

    nx, input_dims  = np.shape(x)
    ny, output_dims = np.shape(y)

    if nx != ny:
        raise ValueError(f"mismatch in number of samples between x and y: {nx} != {ny}")
    n = nx

    train_size = int(train_ratio * n)
    nbatches = (train_size + batch_size - 1) // batch_size
    train_size = nbatches * batch_size

    val_size = n - train_size

    seed = 12345
    x_train, x_valid = random_split(x, [train_size, val_size], seed)
    y_train, y_valid = random_split(y, [train_size, val_size], seed)


    print(f"input dimensions {input_dims}, output dimensions {output_dims}")
    print(f"training set size: {len(x_train)} ({nbatches} batches)")
    print(f"validation set size: {len(x_valid)}")

    data_set = TensorDataset(x_train, y_train)
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=True)

    model = MLP(input_dims, output_dims, hidden_layers)
    model.apply(init_weights)

    model, train_losses, valid_losses = train_model(model,
                                                    data_loader,
                                                    x_valid,
                                                    y_valid,
                                                    lr=0.1,
                                                    #device=torch.device('cuda'),
                                                    info_every=10)

    if plot_errors:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        epochs = np.arange(len(train_losses))
        ax.plot(epochs, train_losses, '-', label='training loss')
        ax.plot(epochs, valid_losses, '-', label='validation loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_yscale('log')
        ax.legend()
        plt.show()

    if output_path is not None:
        save_model_states(model,
                          xshift=xshift,
                          xscale=xscale,
                          yshift=yshift,
                          yscale=yscale,
                          path=output_path)

    return train_losses[-1], valid_losses[-1]



def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Train neural network to build a surrogate.')
    parser.add_argument('csv', type=str, help="The data to train on.")
    parser.add_argument('--output-dir', type=str, required=True, help="Where to save the trained surrogate states.")
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    train(df=df,
          input_var=["v", "FvK"],
          output_var=["eq_D", "eq_hmin", "eq_hmax"],
          output_path=os.path.join(output_dir, "eq.pkl"),
          hidden_layers = 3 * [32],
          batch_size=256)

    train(df=df,
          input_var=["v", "mu", "FvK", "b2", "Fext"],
          output_var=["stretch_D0", "stretch_D1"],
          output_path=os.path.join(output_dir, "stretch.pkl"),
          hidden_layers = 3 * [32])

    df_tc = pd.DataFrame({"v": df["v"],
                          "mu": df["mu"],
                          "FvK": df["FvK"],
                          "b2": df["b2"],
                          "etam": df["etam"],
                          "tc_": df["relax_tc"] * df["mu"] / df["etam"]})

    train(df=df_tc,
          input_var=["v", "mu", "FvK", "b2", "etam"],
          output_var=["tc_"],
          output_path=os.path.join(output_dir, "relax.pkl"),
          hidden_layers = 3 * [32])


if __name__ == "__main__":
    main(sys.argv[1:])
