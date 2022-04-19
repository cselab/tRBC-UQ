#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_file: str):
    df = pd.read_csv(csv_file)
    df.sort_values(by="shear_rate", inplace=True)
    shear_rate = df['shear_rate'].to_numpy()
    theta = df['theta'].to_numpy()
    ttf = df['ttf'].to_numpy()

    fix, axes = plt.subplots(nrows=1, ncols=2)

    ax = axes[0]
    ax.plot(shear_rate, ttf, '-ok')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\dot{\gamma} [s^{-1}]$')
    ax.set_ylabel(r'$\nu / (4 \pi \dot{\gamma})$')

    ax = axes[1]
    ax.plot(shear_rate, theta * 180 / np.pi, '-ok')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\dot{\gamma} [s^{-1}]$')
    ax.set_ylabel(r'$\theta$')

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot inclination angle and ttf against the shear rate")
    parser.add_argument('csv_file', type=str, help="The processed results to plot.")
    args = parser.parse_args()

    plot(args.csv_file)
