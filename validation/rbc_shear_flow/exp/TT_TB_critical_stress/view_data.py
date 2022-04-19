#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_name: str):
    df = pd.read_csv(csv_name)
    eta = df['eta'].to_numpy()
    tau = df['tau'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(eta, tau, 'ok')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r"$\eta$ [Pa.s]")
    ax.set_ylabel(r"$\tau$ [Pa]")

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help="Experimental data.")
    args = parser.parse_args()

    plot(args.csv)
