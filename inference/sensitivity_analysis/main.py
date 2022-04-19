#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

try:
    from tqdm import trange
except(ImportError):
    print("tqdm module not found, install it to see a progress bar during preprocessing.")
    trange = range

from SALib.sample import saltelli
from SALib.analyze import sobol

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from prior import (VarConfig,
                   comp_variables_dict)

sys.path.insert(0, os.path.join(here, "..", "surrogate"))
sys.path.insert(0, os.path.join(here, "..", "surrogate", 'nn_surrogate')) # because of pickle

from nn_surrogate import Surrogate


def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('surr_dir', type=str, help="The directory that contains the trained states of the NN.")
    parser.add_argument('--out', type=str, default=None, help="The resulting csv file.")
    parser.add_argument('--show-plot', action='store_true', default=False, help="If set, plot the figure.")
    args = parser.parse_args(argv)

    surrogate = Surrogate(args.surr_dir)

    comp_variables = [
        comp_variables_dict['v'],
        comp_variables_dict['mu'],
        comp_variables_dict['FvK'],
        comp_variables_dict['b2'],
        comp_variables_dict['etam']
    ]

    nvars = len(comp_variables)

    problem = {
        'num_vars': nvars,
        'names': [var.name for var in comp_variables],
        'bounds': [[var.low(), var.high()] for var in comp_variables]
    }

    # sample
    param_values = saltelli.sample(problem, 1024)

    # evaluate
    y = []
    nFext = 100
    Fext = np.linspace(0, 200, nFext)

    print("Evaluating the model on the sample points")
    for i in trange(len(param_values)):
        v, mu, FvK, b2, etam = param_values[i]
        D, hmin, hmax = surrogate.evaluate_equilibrium([v, mu, FvK, b2])
        D0, D1 = surrogate.evaluate_stretching([v, mu, FvK, b2], Fext.tolist())
        tc = surrogate.evaluate_relax([v, mu, FvK, b2, etam])
        y += [[D, hmin, hmax, tc] + np.array(D0).tolist() + np.array(D1).tolist()]
    y = np.array(y)

    # analyse
    sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]

    S1s = np.array([s['S1'] for s in sobol_indices])

    obsnames = [r"$D$", r"$h_{min}$", r"$h_{max}$", r"$t_c$", r"$D_0$", r"$D_1$"]

    nobs = len(obsnames)
    n0 = 4 # D, hmin, hmax, tc

    if args.show_plot:
        fig, axes = plt.subplots(nrows=1,
                                 ncols=nobs,
                                 figsize=(15, 6),
                                 sharey='col',
                                 gridspec_kw = {'wspace':0, 'hspace':0})

        for i, ax in enumerate(axes):
            if i == 0:
                ax.set_ylabel("First-order Sobol index")
            else:
                ax.set_yticklabels([])
                ax.set_ylim(0, 1)
                ax.set_title(obsnames[i])

            if i < n0:
                ax.bar(x=range(nvars), height=S1s[i, :])
                ax.set_xticks(range(nvars))
                ax.set_xticklabels([var.name for var in comp_variables])
            else:
                start = n0 + (i-n0) * nFext
                end = start + nFext
                for j in range(nvars):
                    ax.plot(Fext, S1s[start:end,j], label=comp_variables[j].name)
                ax.set_xlabel(r"$F_{ext}$")
                ax.legend()

        plt.show()

    out = args.out
    if out is not None:
        data = {'variables': [var.name for var in comp_variables],
                'D': S1s[0,:],
                'hmin': S1s[1,:],
                'hmax': S1s[2,:],
                'tc': S1s[3,:]}
        for j in range(nFext):
            data[f"D0_{j}"] = S1s[n0+j,:]
            data[f"D1_{j}"] = S1s[n0+nFext+j,:]

        pd.DataFrame(data).to_csv(out, index=False)



if __name__ == '__main__':
    main(sys.argv[1:])
