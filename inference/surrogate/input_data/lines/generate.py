#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import sys

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", '..'))

from prior import surrogate_variables_dict


def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--along', type=str, required=True, help="variable that will vary.")
    parser.add_argument('--nsamples', type=int, default=8, help="Number of samples.")
    parser.add_argument('--out', type=str, default="samples.csv", help="output csv.")
    args = parser.parse_args(argv)

    nsamples = args.nsamples

    defaults_vals = {
        'v': 0.95,
        'mu': 4,
        'FvK': 250,
        'b2': 1.5,
        'etam': 0.7,
        'Fext': 100
    }

    along = args.along
    assert along in defaults_vals.keys()

    data = {}
    for name in surrogate_variables_dict.keys():
        var = surrogate_variables_dict[name]

        if name == along:
            data[name] = np.linspace(var.low(), var.high(), nsamples)
        else:
            data[name] = np.full(nsamples, defaults_vals[name])

        data[name] = (data[name] - var.low()) / (var.high() - var.low())

    df = pd.DataFrame(data)
    df.to_csv(args.out, index=False)




if __name__ == '__main__':
    main(sys.argv[1:])
