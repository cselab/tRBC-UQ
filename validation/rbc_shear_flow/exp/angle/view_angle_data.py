#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_name: str):
    df = pd.read_csv(csv_name)
    shear_rate = df['shear_rate'].to_numpy()

    ndonors = len(df.columns)-1

    fig, ax = plt.subplots()

    for i in range(ndonors):
        theta = df[f'angle{i+1}'].to_numpy()
        ax.plot(shear_rate, theta, '+')

    ax.set_xscale('log')
    ax.set_xlim(0.1,1000)
    ax.set_ylim(0,25)

    ax.set_xlabel(r"$\dot{\gamma} [s^{-1}]$")
    ax.set_ylabel(r"$\theta$")

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot the orientation angle against the  shear rate")
    parser.add_argument('csv', type=str, help="The angle data.")
    args = parser.parse_args()

    plot(args.csv)
