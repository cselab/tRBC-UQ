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
    parser = argparse.ArgumentParser(description='Forward propagation on stretching.')
    parser.add_argument('surr_dir', type=str, help="The directory containing the trained surrogate state.")
    parser.add_argument('samples_file', type=str, help="The posterior distribution of the parameters (csv file, see korali_to_csv.py).")
    parser.add_argument('--output-csv', type=str, default="forward_stretching.csv", help="The output file containings mean and quantiles of diameters against Fext.")
    args = parser.parse_args(argv)

    nFext = 100
    Fext = np.linspace(0, 200, nFext)

    surrogate = Surrogate(args.surr_dir)
    df = pd.read_csv(args.samples_file)

    all_v    = df['v'].to_numpy()
    all_mu   = df['mu'].to_numpy()
    all_kb   = df['kb'].to_numpy()
    all_b2   = df['b2'].to_numpy()
    all_sig0 = df['sigma0'].to_numpy()
    all_sig1 = df['sigma1'].to_numpy()
    RA = np.sqrt(135/(4*np.pi))

    nsamples = len(all_mu)

    all_D0 = np.zeros((nFext, nsamples))
    all_D1 = np.zeros((nFext, nsamples))

    print(f"Evaluate stretching from samples {args.samples_file}")
    for i in trange(nsamples):
        v = all_v[i]
        mu = all_mu[i]
        kb = all_kb[i]
        b2 = all_b2[i]
        FvK = mu * RA**2 / kb
        D0, D1 = surrogate.evaluate_stretching([v, mu, FvK, b2], Fext)
        D0 = np.array(D0)
        D1 = np.array(D1)

        all_D0[:,i] = D0
        all_D1[:,i] = D1

    # compute confidence intervals:
    # discretize the axis, compute CDF and invert at percentiles
    n = 1000
    yD0 = np.linspace(0, 1.5 * np.max(all_D1), n)
    yD1 = np.linspace(0, 1.5 * np.max(all_D1), n)

    cdf_D0 = np.zeros((nFext, n))
    cdf_D1 = np.zeros((nFext, n))

    print("Computing CDF on a grid:")
    for i in trange(n):
        cdf_D0[:,i] = np.sum(norm.cdf(yD0[i], loc=all_D0, scale=all_sig0[np.newaxis,:]), axis=1) / nsamples
        cdf_D1[:,i] = np.sum(norm.cdf(yD1[i], loc=all_D1, scale=all_sig1[np.newaxis,:]), axis=1) / nsamples

    credible_intervals = [0.50, 0.75, 0.90, 0.99]
    quantiles = []
    for c in credible_intervals:
        q = (1-c)/2
        quantiles += [q, 1-q]

    data = {}
    data["Fext"] = Fext

    for q in quantiles:
        q_D0 = yD0[np.argmin(np.abs(cdf_D0 - q), axis=1)]
        q_D1 = yD1[np.argmin(np.abs(cdf_D1 - q), axis=1)]

        data[f"D0_q_{q:.3}"] = q_D0
        data[f"D1_q_{q:.3}"] = q_D1

    data["D0_mean"] = np.mean(all_D0, axis=1)
    data["D1_mean"] = np.mean(all_D1, axis=1)

    df = pd.DataFrame(data)
    df.to_csv(args.output_csv, index=False)




if __name__ == "__main__":
    main(sys.argv[1:])
