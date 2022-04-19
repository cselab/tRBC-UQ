#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import pathlib
import sys
import torch

from utils import TemporaryNumpySeed

sys.path.append('..')
from nn_surrogate import Surrogate

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


def compute_validation_loss(s: Surrogate,
                            df: pd.DataFrame,
                            input_var: list,
                            output_var: list,
                            batch_size: int=128,
                            train_ratio: float=0.8):

    nsamples, _ = df.shape

    x = torch.zeros((nsamples, len(input_var )))
    y = torch.zeros((nsamples, len(output_var)))

    for i, name in enumerate(input_var):
        data = df[name].to_numpy()
        x[:,i] = torch.from_numpy(data)

    for i, name in enumerate(output_var):
        data = df[name].to_numpy()
        y[:,i] = torch.from_numpy(data)

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


    sy = np.zeros_like(y_valid)

    for i, x in enumerate(x_valid):
        v, mu, FvK, b2, etam, Fext = x.tolist()

        D, hmin, hmax = s.evaluate_equilibrium([v, mu, FvK, b2])
        D0, D1 = s.evaluate_stretching([v, mu, FvK, b2], [Fext])
        D0, D1 = D0[0], D1[0]
        tc = s.evaluate_relax([v, mu, FvK, b2, etam])
        sy[i,:] = [D, hmin, hmax, D0, D1, tc]


    for i, varname in enumerate(output_var):
        cur = sy[:,i]
        ref = np.array(y_valid[:,i])
        dy = cur - ref
        mse = np.mean(dy**2)
        mL2 = np.sqrt(mse)
        mL1 = np.mean(np.abs(dy))
        print(varname)
        print(f"MSE: {mse}")
        print(f"mL2: {mL2}")
        print(f"mL1: {mL1}")
        print(f"mean relative deviation: {100 * np.mean(np.abs(dy) / np.abs(ref))}%")
        print()



def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help="The samples data.")
    parser.add_argument('surrogate_dir', type=str, help="The trained surrogate directory.")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    s = Surrogate(args.surrogate_dir)

    compute_validation_loss(s, df,
                            input_var = ["v", "mu", "FvK", "b2", "etam", "Fext"],
                            output_var = ["eq_D", "eq_hmin", "eq_hmax",
                                          "stretch_D0", "stretch_D1",
                                          "relax_tc"])


if __name__ == "__main__":
    main(sys.argv[1:])
