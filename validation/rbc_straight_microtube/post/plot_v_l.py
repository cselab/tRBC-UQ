#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Plot rbc length against the rbc velocity.')
    parser.add_argument('sim_csv', type=str, help="The simulation data.")
    parser.add_argument('--exp-csv', type=str, default="exp/Tomaiuolo2009_v_l_6.6um.csv", help="The experimental data.")
    args = parser.parse_args(argv)

    df_exp = pd.read_csv(args.exp_csv)
    df_exp.sort_values(by='v', inplace=True)

    df_sim = pd.read_csv(args.sim_csv)
    df_sim.sort_values(by='v', inplace=True)

    L_exp = df_exp['l'].to_numpy()
    v_exp = df_exp['v'].to_numpy()

    L_sim = df_sim['l'].to_numpy()
    v_sim = df_sim['v'].to_numpy()

    fig, ax = plt.subplots()

    if "l_high" in df_exp.columns:
        dL_exp = (df_exp['l_high'].to_numpy() - df_exp['l_low'].to_numpy())/2
        ax.errorbar(v_exp, L_exp, yerr=dL_exp, fmt='ok', capsize=2, label='Experiments')
    else:
        ax.plot(v_exp, L_exp, 'ok', label='Experiments')

    ax.plot(v_sim, L_sim, '-or', label='Simulations')

    ax.set_xlabel(r'$v [cm/s]$')
    ax.set_ylabel(r'$l [um]$')

    # ax.set_xlim(1, 2000)
    ax.set_ylim(4, 12)
    ax.legend()
    ax.set_xscale('log')

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
