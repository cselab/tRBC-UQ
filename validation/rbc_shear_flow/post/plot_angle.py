#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(sim_csv_fname: str,
         exp_csv_fname: str):

    fix, ax = plt.subplots()

    # exp data
    df = pd.read_csv(exp_csv_fname)
    ndonors = len(df.columns)-1
    shear_rate = df['shear_rate'].to_numpy()

    for i in range(ndonors):
        theta = df[f'angle{i+1}'].to_numpy()
        ax.plot(shear_rate, theta, 'vk')


    # sim data
    df = pd.read_csv(sim_csv_fname)
    df.sort_values(by="shear_rate", inplace=True)
    shear_rate = df['shear_rate'].to_numpy()
    theta = df['theta'].to_numpy()
    ax.plot(shear_rate, theta * 180 / np.pi, '-or')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\dot{\gamma} [s^{-1}]$')
    ax.set_ylabel(r'$\theta$')

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot inclination angle against the shear rate")
    parser.add_argument('sim_csv_fname', type=str, help="The simulation results.")
    parser.add_argument('exp_csv_fname', type=str, help="The experimental data.")
    args = parser.parse_args()

    plot(args.sim_csv_fname, args.exp_csv_fname)
