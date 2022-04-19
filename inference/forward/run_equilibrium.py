#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import sys
from scipy.stats import norm

try:
    from tqdm import trange
except(ImportError):
    print("tqdm module not found, install it to see a progress bar during preprocessing.")
    trange = range

here = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(here, "..", "surrogate"))
sys.path.insert(0, os.path.join(here, "..", "surrogate", "nn_surrogate")) # because of pickle

from nn_surrogate import Surrogate

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Forward propagation on equilibrium shape.')
    parser.add_argument('surr_dir', type=str, help="The directory containing the trained surrogate state.")
    parser.add_argument('samples_file', type=str, help="The posterior distribution of the parameters (csv file, see korali_to_csv.py).")
    parser.add_argument('--output-csv', type=str, default="forward_eq.csv", help="The output file containing the pdfs of hmin, hmax and D.")
    args = parser.parse_args(argv)

    surrogate = Surrogate(args.surr_dir)
    df = pd.read_csv(args.samples_file)

    all_v   = df['v'].to_numpy()
    all_FvK = df['FvK'].to_numpy()
    all_sig = df['sigma'].to_numpy()

    nsamples = len(all_v)
    all_D = np.zeros(nsamples)
    all_hmin = np.zeros(nsamples)
    all_hmax = np.zeros(nsamples)

    print(f"Evaluate stretching from samples {args.samples_file}")
    for i in trange(nsamples):
        v = all_v[i]
        mu = 4 # arbitrary
        FvK = all_FvK[i]
        b2 = 2 # arbitrary

        D, hmin, hmax = surrogate.evaluate_equilibrium([v, mu, FvK, b2])
        all_D[i] = D
        all_hmin[i] = hmin
        all_hmax[i] = hmax

    all_sigD = all_D * all_sig
    all_sighmin = all_hmin * all_sig
    all_sighmax = all_hmax * all_sig

    # compute pdf of each measurement on discrete axis
    n = 200
    ns = 3

    muD = np.mean(all_D)
    dD = ns * (np.std(all_D) + np.mean(all_sigD))

    muhmin = np.mean(all_hmin)
    dhmin = ns * (np.std(all_hmin) + np.mean(all_sighmin))

    muhmax = np.mean(all_hmax)
    dhmax = ns * (np.std(all_hmax) + np.mean(all_sighmax))

    yD = np.linspace(muD - dD, muD + dD, n)
    yhmin = np.linspace(muhmin - dhmin, muhmin + dhmin, n)
    yhmax = np.linspace(muhmax - dhmax, muhmax + dhmax, n)

    pdf_D = np.sum( [norm.pdf(y_, loc=all_D, scale=all_sigD) for y_ in yD], axis=1 ) / nsamples
    pdf_hmin = np.sum( [norm.pdf(y_, loc=all_hmin, scale=all_sighmin) for y_ in yhmin], axis=1 ) / nsamples
    pdf_hmax = np.sum( [norm.pdf(y_, loc=all_hmax, scale=all_sighmax) for y_ in yhmax], axis=1 ) / nsamples


    df = pd.DataFrame({"D": yD,
                       "pdf_D": pdf_D,
                       "hmin": yhmin,
                       "pdf_hmin": pdf_hmin,
                       "hmax": yhmax,
                       "pdf_hmax": pdf_hmax})
    df.to_csv(args.output_csv, index=False)



if __name__ == "__main__":
    main(sys.argv[1:])
