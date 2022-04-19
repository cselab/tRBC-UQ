#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

from utils import get_samples


def plot(korali_file: str,
         key: str,
         nbins: int=50):
    samples = get_samples(korali_file, extract_likelihood=True)

    log_likelihood = samples['log_likelihood']

    if not key in samples and  key == 'kb':
        RAsq = 135 / (4 * np.pi)
        kb = samples['mu'] * RAsq / samples['FvK']
        values = kb
    else:
        values = samples[key]

    fig, axsamples = plt.subplots()
    axprofile = axsamples.twinx()

    # plot raw samples
    axsamples.hist(values, bins=nbins, density=True)
    axsamples.set_xlabel(key)
    axsamples.set_ylabel(f"p(key)")

    # plot profile likelihood
    color = 'tab:red'
    statistic, bin_edges, _ = binned_statistic(x=values,
                                               values=log_likelihood,
                                               statistic='max',
                                               bins=nbins)

    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    axprofile.plot(x, statistic, color=color)
    axprofile.set_ylabel(r"$\log{L}$")
    axprofile.tick_params(axis='y', labelcolor=color)

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('korali_file', type=str, help="The samples produced by korali.")
    parser.add_argument('var_name', type=str, help="The variable name to plot.")
    args = parser.parse_args()

    plot(korali_file=args.korali_file,
         key=args.var_name)
