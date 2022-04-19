#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import sys

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Plot forward propagation of stretching.')
    parser.add_argument('stats', type=str, help="The csv file containing the propagation statistics.")
    parser.add_argument('--exp', type=str, nargs='+', default=[], help="The csv file(s) containing the experimental data.")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.stats)

    Fext = df['Fext'].to_numpy()
    D0 = df['D0_mean'].to_numpy()
    D1 = df['D1_mean'].to_numpy()

    fig, ax = plt.subplots()

    ax.plot(Fext, D0, '--k')
    ax.plot(Fext, D1, '--k')

    color = 'cornflowerblue'
    alpha = 0.15

    credible_intervals = [0.50, 0.75, 0.90, 0.99]
    for c in credible_intervals:
        qm = (1-c)/2
        qp = 1 - qm
        D0p = df[f"D0_q_{qp:.3}"].to_numpy()
        D0m = df[f"D0_q_{qm:.3}"].to_numpy()

        ax.fill_between(Fext, D0m, D0p, alpha=alpha, color=color)

        D1p = df[f"D1_q_{qp:.3}"].to_numpy()
        D1m = df[f"D1_q_{qm:.3}"].to_numpy()

        ax.fill_between(Fext, D1m, D1p, alpha=alpha, color=color)

    for fexp in args.exp:
        df = pd.read_csv(fexp)
        Fext = df['Fext'].to_numpy()
        D0 = df['D0'].to_numpy()
        D1 = df['D1'].to_numpy()

        ax.plot(Fext, D0, 'ok')
        ax.plot(Fext, D1, 'ok')

    ax.set_xlim(0, 200)
    ax.set_xlabel(r"$F_{ext} [pN]$")
    ax.set_ylabel(r"$D [\mu m]$")

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
