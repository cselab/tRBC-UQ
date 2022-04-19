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
    parser = argparse.ArgumentParser(description="Draw samples uniformly in the parameters space.")
    parser.add_argument('--nsamples', type=int, default=1000, help="Number of samples.")
    parser.add_argument('--out', type=str, default="samples.csv", help="output csv.")
    args = parser.parse_args(argv)

    nsamples = args.nsamples

    data = {}
    for name in surrogate_variables_dict.keys():
        var = surrogate_variables_dict[name]
        data[name] = np.random.uniform(var.low(), var.high(), nsamples)
        data[name] = (data[name] - var.low()) / (var.high() - var.low())

    df = pd.DataFrame(data)
    df.to_csv(args.out, index=False)




if __name__ == '__main__':
    main(sys.argv[1:])
