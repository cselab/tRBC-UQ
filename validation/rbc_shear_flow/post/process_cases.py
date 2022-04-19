#!/usr/bin/env python

desc = """
Compute the inclination angle and the dimensionless ttf of a red blood cell in shear flow for many simulations.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import re
import sys

from compute_angle import compute_mean_inclination_angle
from compute_ttf import compute_dimensionless_ttf

# because of pickle
here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from parameters import RBCShearParams


def get_param(basedir: str, param_name: str):
    rexf = '[-+]?\d*\.\d+|\d+'
    matches = re.findall(f"{param_name}_({rexf})", basedir)
    assert len(matches) == 1
    return float(matches[0])

def main(basedir: str,
         against: str,
         out: str):

    simdirs = sorted(glob.glob(os.path.join(basedir, "*")))
    params = []
    thetas = []
    dthetas = []
    ttfs = []

    assert against != 'theta'

    for simdir in simdirs:
        theta, dtheta = compute_mean_inclination_angle(simdir)
        try:
            ttf = compute_dimensionless_ttf(simdir)
        except RuntimeError:
            ttf = float("NAN")
        param = get_param(simdir, against)
        params += [param]
        thetas += [theta]
        dthetas += [dtheta]
        ttfs += [ttf]
        print(f"{against} = {param}, theta = {180*theta/np.pi}, ttf = {ttf}")

    if out is not None:
        data = {against: params,
                "theta": thetas,
                "dtheta": dthetas,
                "ttf": ttfs}
        df = pd.DataFrame(data)
        df.sort_values(by=against, inplace=True)
        df.to_csv(out, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('basedir', type=str, help="The directory containing all simulation results.")
    parser.add_argument('--against', type=str, default='shear_rate', help="The input parameter name.")
    parser.add_argument('--out', type=str, default=None, help="The output csv file.")
    args = parser.parse_args()

    main(args.basedir, args.against, args.out)
