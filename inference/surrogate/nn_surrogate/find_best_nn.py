#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys

from train import train

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Perform a grid search to find the best NN architecture.')
    parser.add_argument('csv', type=str, help="The data to train on.")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)

    depths = []
    layers = []
    valid_losses = []

    for depth in [1, 2, 3, 4]:
        for layer in [8, 16, 32, 48, 64, 96, 128, 256]:
            _, err = train(df=df,
                           input_var=["v", "FvK"],
                           output_var=["eq_D", "eq_hmin", "eq_hmax"],
                           output_path=None,
                           hidden_layers = depth * [layer])
            depths += [depth]
            layers += [layer]
            valid_losses += [err]
            print(depth, layer, err)

    for d, l, err in zip(depths, layers, valid_losses):
        print(d, l, err)

    i = np.argmin(valid_losses)
    print(f"Best found: {depths[i]} X {layers[i]} -> {valid_losses[i]}")

if __name__ == "__main__":
    main(sys.argv[1:])
