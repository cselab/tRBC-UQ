#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Plot forward propagation of relaxation.')
    parser.add_argument('stats', type=str, help="The csv file containing the propagation statistics.")
    parser.add_argument('--exp', type=str, nargs='+', default=[], help="The csv file(s) containing the experimental data.")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.stats)

    time = df['time'].to_numpy()
    LW = df['z_mean'].to_numpy()

    fig, ax = plt.subplots()

    ax.plot(time, LW, '--k')

    color = 'cornflowerblue'
    alpha = 0.15

    credible_intervals = [0.50, 0.75, 0.90, 0.99]
    for c in credible_intervals:
        qm = (1-c)/2
        qp = 1 - qm
        LWp = df[f"z_q_{qp:.3}"].to_numpy()
        LWm = df[f"z_q_{qm:.3}"].to_numpy()

        ax.fill_between(time, LWm, LWp, alpha=alpha, color=color)

    for fexp in args.exp:
        df = pd.read_csv(fexp)
        t = df['t'].to_numpy()
        LW = df['L_W'].to_numpy()

        ax.plot(t, LW, 'ok')

    ax.set_xlim(0,)
    ax.set_xlabel(r"$t [s]$")
    ax.set_ylabel(r"$L/W$")

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
