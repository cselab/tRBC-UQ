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

sys.path.insert(0, os.path.join(here, "..", "inference"))

from common import exponential_model

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Forward propagation on equilibrium shape.')
    parser.add_argument('samples_file', type=str, help="The posterior distribution of the parameters (csv file, see korali_to_csv.py).")
    parser.add_argument('--output-csv', type=str, default="forward_rel.csv", help="The output file containing the pdfs of tc.")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.samples_file)
    all_tc    = df['tc'].to_numpy()
    all_z_0   = df['z0'].to_numpy()
    all_z_inf = df['zinf'].to_numpy()
    all_sig_z = df['sigma_z'].to_numpy()

    nsamples = len(all_sig_z)

    ntime = 100
    time = np.linspace(0, 0.5, ntime)

    all_z = np.zeros((ntime, nsamples))

    print(f"Evaluate z(t) from samples {args.samples_file}")
    for i in trange(nsamples):
        tc = all_tc[i]
        z_0 = all_z_0[i]
        z_inf = all_z_inf[i]
        z = exponential_model(z0=z_0, zinf=z_inf, tc=tc, t=time)
        all_z[:,i] = z


    # compute confidence intervals:
    # discretize the axis, compute CDF and invert at percentiles
    n = 1000
    y_z = np.linspace(0, 1.5 * np.max(all_z), n)
    cdf_z = np.zeros((ntime, n))

    print("Computing CDF on a grid:")
    for i in trange(n):
        cdf_z[:,i] = np.sum(norm.cdf(y_z[i], loc=all_z, scale=all_sig_z[np.newaxis,:]), axis=1) / nsamples

    credible_intervals = [0.50, 0.75, 0.90, 0.99]
    quantiles = []
    for c in credible_intervals:
        q = (1-c)/2
        quantiles += [q, 1-q]

    data = {}
    data["time"] = time

    for q in quantiles:
        q_z = y_z[np.argmin(np.abs(cdf_z - q), axis=1)]
        data[f"z_q_{q:.3}"] = q_z

    data["z_mean"] = np.mean(all_z, axis=1)

    pd.DataFrame(data).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
