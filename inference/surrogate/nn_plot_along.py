#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from nn_surrogate import Surrogate

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.path.insert(0, os.path.join(here, "nn_surrogate"))

from prior import surrogate_variables_dict

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('surr_dir', type=str, help="Directory containing the trained surrogate states.")
    parser.add_argument('samples', type=str, help="Evaluated samples.")
    parser.add_argument('--along', type=str, required=True, help="variable that varies.")
    args = parser.parse_args(argv)

    s = Surrogate(args.surr_dir)

    along = args.along

    df = pd.read_csv(args.samples)

    x = df[along].to_numpy()
    idx = np.argsort(x)
    x = x[idx]

    v    = df['v'].to_numpy()[idx]
    mu   = df['mu'].to_numpy()[idx]
    FvK  = df['FvK'].to_numpy()[idx]
    b2   = df['b2'].to_numpy()[idx]
    etam = df['etam'].to_numpy()[idx]
    Fext = df['Fext'].to_numpy()[idx]

    sD = np.zeros_like(v)
    shmin = np.zeros_like(v)
    shmax = np.zeros_like(v)
    sD0 = np.zeros_like(v)
    sD1 = np.zeros_like(v)
    stc = np.zeros_like(v)

    for i in range(len(v)):
        p = [v[i], mu[i], FvK[i], b2[i]]
        sD[i], shmin[i], shmax[i] = s.evaluate_equilibrium(p)
        _D0, _D1 = s.evaluate_stretching(p, [Fext[i]])
        sD0[i], sD1[i] = _D0[0], _D1[0]
        stc[i] = s.evaluate_relax(p + [etam[i]])

    D = df['eq_D'].to_numpy()[idx]
    hmin = df['eq_hmin'].to_numpy()[idx]
    hmax = df['eq_hmax'].to_numpy()[idx]
    D0 = df['stretch_D0'].to_numpy()[idx]
    D1 = df['stretch_D1'].to_numpy()[idx]
    tc = df['relax_tc'].to_numpy()[idx]


    fig, axes = plt.subplots(2,3)
    fmt='+k'
    sfmt='-r'

    ax = axes[0,0]
    ax.plot(x, D, fmt)
    ax.plot(x, sD, sfmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$D$')

    ax = axes[0,1]
    ax.plot(x, hmin, fmt)
    ax.plot(x, shmin, sfmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$h_{min}$')

    ax = axes[0,2]
    ax.plot(x, hmax, fmt)
    ax.plot(x, shmax, sfmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$h_{max}$')

    ax = axes[1,0]
    ax.plot(x, D0, fmt)
    ax.plot(x, sD0, sfmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$D_0$')

    ax = axes[1,1]
    ax.plot(x, D1, fmt)
    ax.plot(x, sD1, sfmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$D_1$')

    ax = axes[1,2]
    ax.plot(x, tc, fmt)
    ax.plot(x, stc, sfmt)
    ax.set_xlabel(along)
    ax.set_ylabel(r'$t_c$')

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
