#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Plot relative viscosity against the diameter of the pipe.')
    parser.add_argument('--exp-csv', type=str, default="exp/Tomaiuolo2009_pg_v_6.6um.csv", help="The experimental data.")
    parser.add_argument('--sim-csv', type=str, default="results/pg_v_6.6um.csv", help="The simulation data.")
    args = parser.parse_args(argv)

    df_exp = pd.read_csv(args.exp_csv)
    df_sim = pd.read_csv(args.sim_csv)

    pg_exp = df_exp['pg'].to_numpy()
    v_exp = df_exp['v'].to_numpy()

    pg_sim = df_sim['pg'].to_numpy()
    v_sim = df_sim['v'].to_numpy()

    idx = np.argsort(pg_sim)
    pg_sim = pg_sim[idx]
    v_sim = v_sim[idx]


    fig, ax = plt.subplots()

    ax.plot(pg_exp, v_exp, 'ok', label='Experiments')
    ax.plot(pg_sim, v_sim, '-or', label='Simulations')

    ax.set_xlabel(r'$\Delta p / L [mmHg / mm]$')
    ax.set_ylabel(r'$v [cm/s]$')

    # ax.set_xlim(1, 2000)
    # ax.set_ylim(1, 5)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
