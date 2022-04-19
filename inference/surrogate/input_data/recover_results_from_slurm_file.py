#!/usr/bin/env python

import numpy as np
import pandas as pd

def parse(fname: str):

    keys = ['v', 'mu', 'FvK', 'b2', 'etam', 'Fext',
            'eq_D', 'eq_hmin', 'eq_hmax', 'stretch_D0', 'stretch_D1', 'relax_tc']

    all_vals = []

    with open(fname) as f:
        for line in f.readlines():
            try:
                vals = [float(val) for val in line.split(' ')]
            except:
                pass
            else:
                assert len(vals) == len(keys)
                all_vals += [vals]

    all_vals = np.array(all_vals)
    data = {key: all_vals[:,i] for i, key in enumerate(keys)}
    return data



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', type=str, help="Slurm file.")
    parser.add_argument('out', type=str, help="Result csv file.")
    args = parser.parse_args()

    data = parse(args.fname)
    pd.DataFrame(data).to_csv(args.out, index=False)
