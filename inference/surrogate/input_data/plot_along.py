#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('samples', type=str, help="Evaluated samples.")
    parser.add_argument('--along', type=str, required=True, help="variable that varies.")
    args = parser.parse_args(argv)

    along = args.along

    df = pd.read_csv(args.samples)

    x = df[along].to_numpy()
    idx = np.argsort(x)
    x = x[idx]

    D = df['eq_D'].to_numpy()[idx]
    hmin = df['eq_hmin'].to_numpy()[idx]
    hmax = df['eq_hmax'].to_numpy()[idx]
    D0 = df['stretch_D0'].to_numpy()[idx]
    D1 = df['stretch_D1'].to_numpy()[idx]
    tc = df['relax_tc'].to_numpy()[idx]

    fig, axes = plt.subplots(ncols=3, nrows=2)
    fmt='-+k'

    ax = axes[0,0]
    ax.plot(x, D, fmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$D$')

    ax = axes[0,1]
    ax.plot(x, hmin, fmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$h_{min}$')

    ax = axes[0,2]
    ax.plot(x, hmax, fmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$h_{max}$')

    ax = axes[1,0]
    ax.plot(x, D0, fmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$D0$')

    ax = axes[1,1]
    ax.plot(x, D1, fmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$D1$')

    ax = axes[1,2]
    ax.plot(x, tc, fmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$t_c$')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
