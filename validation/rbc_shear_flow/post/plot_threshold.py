#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(sim_csv_name: str,
         exp_csv_names: list[str]):
    fig, ax = plt.subplots()

    exp_kw_args = [
        {"marker": "o",
         "markerfacecolor": "none",
         "markeredgecolor": "k"},
        {"marker": "o",
         "markerfacecolor": "k",
         "markeredgecolor": "k"}
    ]

    # exp
    for i, exp_csv_name in enumerate(exp_csv_names):
        df = pd.read_csv(exp_csv_name)
        eta = df['eta'].to_numpy()
        tau = df['tau'].to_numpy()
        ax.plot(eta, tau, **exp_kw_args[i%len(exp_kw_args)], linestyle="none")

    # sim
    df = pd.read_csv(sim_csv_name)
    eta = df['eta'].to_numpy()
    gamma_lo = df['gamma_lo'].to_numpy()
    gamma_hi = df['gamma_hi'].to_numpy()
    tau_lo = eta * gamma_lo * 1e-3
    tau_hi = eta * gamma_hi * 1e-3

    tau = (tau_lo + tau_hi) / 2
    dtau = (tau_hi - tau_lo) / 2
    ax.errorbar(eta, tau, yerr=dtau, color='darkred', linestyle='none', markersize=0, capsize=2)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r"$\eta$ [Pa.s]")
    ax.set_ylabel(r"$\tau$ [Pa]")

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_csv', type=str, help="Simulation data.")
    parser.add_argument('exp_csv', type=str, nargs='+', help="Experimental data.")
    args = parser.parse_args()

    plot(args.sim_csv, args.exp_csv)
