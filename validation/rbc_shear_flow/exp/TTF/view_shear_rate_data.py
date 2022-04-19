#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_name: str):
    df = pd.read_csv(csv_name)
    shear_rate = df['shear_rate'].to_numpy()
    TTF = df['TTF'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(shear_rate, TTF, '+k')

    ax.set_xscale('log')

    ax.set_xlabel(r"$\dot{\gamma} [s^{-1}]$")
    ax.set_ylabel(r"TTF/$\dot{\gamma}$")

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot ttf against =shear rate")
    parser.add_argument('csv', type=str, help="The TTF data.")
    args = parser.parse_args()

    plot(args.csv)
