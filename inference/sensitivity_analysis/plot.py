#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(csv_fname: str):
    df = pd.read_csv(csv_fname)
    var_names = df['variables']

    nFext = (len(df.columns) - 5) // 2 # 5: variables,D,hmin,hmax,tc; 2: D0, D1
    Fext = np.linspace(0, 200, nFext)

    obsnames = [r"$D$", r"$h_{min}$", r"$h_{max}$", r"$t_c$", r"$D_{tr}$", r"$D_{ax}$"]
    varnames = [r"$v$", r"$\mu$", r"FvK", r"$\beta_2$", r"$\eta_m$"]

    nobs = len(obsnames)
    nvars = len(varnames)

    n0 = 4

    S1s = np.zeros((n0 + 2*nFext, nvars))

    S1s[0,:] = df["D"]
    S1s[1,:] = df["hmin"]
    S1s[2,:] = df["hmax"]
    S1s[3,:] = df["tc"]

    for j in range(nFext):
        S1s[n0+j      ,:] = df[f"D0_{j}"]
        S1s[n0+j+nFext,:] = df[f"D1_{j}"]

    fig, axes = plt.subplots(nrows=1,
                             ncols=nobs,
                             figsize=(15, 6),
                             sharey='col',
                             gridspec_kw = {'wspace':0, 'hspace':0})


    for i, ax in enumerate(axes):
        ax.set_title(obsnames[i])
        ax.set_ylim(0, 1)

        if i == 0:
            ax.set_ylabel("First order Sobol index")
        else:
            ax.set_yticklabels([])

        if i < n0:
            ax.bar(x=range(nvars), height=S1s[i, :])
            ax.set_xticks(range(nvars))
            ax.set_xticklabels(varnames)
        else:
            start = n0 + (i-n0) * nFext
            end = start + nFext
            for j in range(nvars):
                ax.plot(Fext, S1s[start:end,j], label=varnames[j])
            ax.set_xlabel(r"$F_{ext} [pN]$")

            if i == len(axes)-1:
                ax.legend()

    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help="The file output by the sensitivity analysis")
    args = parser.parse_args()
    plot(args.csv)

main()
