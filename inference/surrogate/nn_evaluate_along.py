#!/usr/bin/env python

import glob
import numpy as np
import os
import pandas as pd
import sys

from nn_surrogate import Surrogate

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.path.insert(0, os.path.join(here, "nn_surrogate"))

from prior import surrogate_variables_dict

def get_along_name(s: str):
    keys = s.split('_')
    return keys[-2]

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('surr_dir', type=str, help="Directory containing the trained surrogate states.")
    parser.add_argument('samples_dir', type=str, help="Evaluated samples.")
    args = parser.parse_args(argv)

    s = Surrogate(args.surr_dir)

    for samples_filename in glob.glob(os.path.join(args.samples_dir, "along_*_32.csv")):

        along = get_along_name(samples_filename)

        df = pd.read_csv(samples_filename)
        df.sort_values(by=along, inplace=True)

        x = df[along].to_numpy()
        v    = df['v'].to_numpy()
        mu   = df['mu'].to_numpy()
        FvK  = df['FvK'].to_numpy()
        b2   = df['b2'].to_numpy()
        etam = df['etam'].to_numpy()
        Fext = df['Fext'].to_numpy()

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

        D = df['eq_D'].to_numpy()
        hmin = df['eq_hmin'].to_numpy()
        hmax = df['eq_hmax'].to_numpy()
        D0 = df['stretch_D0'].to_numpy()
        D1 = df['stretch_D1'].to_numpy()
        tc = df['relax_tc'].to_numpy()

        data = {along: x,
                "D": D,
                "hmin": hmin,
                "hmax": hmax,
                "D0": D0,
                "D1": D1,
                "tc": tc,
                "sD": sD,
                "shmin": shmin,
                "shmax": shmax,
                "sD0": sD0,
                "sD1": sD1,
                "stc": stc}

        out = f"line_eval_{along}.csv"
        df = pd.DataFrame(data)
        df.sort_values(by=along, inplace=True)
        df.to_csv(out, index=False)

if __name__ == '__main__':
    main(sys.argv[1:])
