#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(csv: str):

    df = pd.read_csv(csv)

    yD = df["D"].to_numpy()
    pdf_D = df["pdf_D"].to_numpy()
    yhmin = df["hmin"].to_numpy()
    pdf_hmin = df["pdf_hmin"].to_numpy()
    yhmax = df["hmax"].to_numpy()
    pdf_hmax = df["pdf_hmax"].to_numpy()


    # experiment data
    muDexp = 7.82
    sigDexp = 0.62
    muhminexp = 0.81
    sighminexp = 0.35
    muhmaxexp = 2.58
    sighmaxexp = 0.27

    fig, axes = plt.subplots(ncols=3, nrows=1)
    ax = axes[0]

    cs=5
    fmt_exp='ok'
    kw_pdf={'alpha': 0.5}

    ax.fill_between(yD, pdf_D, **kw_pdf)
    ax.errorbar(x=[muDexp], y=[np.max(pdf_D)/2],
                xerr=[sigDexp], yerr=None,
                fmt=fmt_exp, capsize=cs)
    ax.set_ylim(0,)
    ax.set_xlabel(r'$D [\mu m]$')

    ax = axes[1]
    ax.fill_between(yhmin, pdf_hmin, **kw_pdf)
    ax.errorbar(x=[muhminexp], y=[np.max(pdf_hmin)/2],
                xerr=[sighminexp], yerr=None,
                fmt=fmt_exp, capsize=cs)

    ax.set_ylim(0,)
    ax.set_xlabel(r'$h_{min} [\mu m]$')

    ax = axes[2]
    ax.fill_between(yhmax, pdf_hmax, **kw_pdf)
    ax.errorbar(x=[muhmaxexp], y=[np.max(pdf_hmax)/2],
                xerr=[sighmaxexp], yerr=None,
                fmt=fmt_exp, capsize=cs)

    ax.set_ylim(0,)
    ax.set_xlabel(r'$h_{max} [\mu m]$')

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot forward propagation of equilibrium.')
    parser.add_argument('csv', type=str, help="The csv file containing the propagation statistics.")
    args = parser.parse_args()

    plot(args.csv)
