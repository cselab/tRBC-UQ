#!/usr/bin/env python

import glob
import numpy as np
import os
import pandas as pd
import sys

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, ".."))
from parameters import (PipeFlowParams,
                        load_parameters)

from compute_rbc_length import get_param, compute_mean_rbc_length_width
from compute_rbc_velocity import compute_mean_rbc_velocity

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=str, help="The directory containing the simulations.")
    parser.add_argument('--out', type=str, default=None, help="output csv file.")
    args = parser.parse_args(argv)

    sim_dirs = glob.glob(os.path.join(args.basedir, "*"))

    all_L = list()
    all_W = list()
    all_v = list()
    all_pg = list()

    for simdir in sim_dirs:
        L, W = compute_mean_rbc_length_width(simdir)
        v = compute_mean_rbc_velocity(simdir).to('cm/s').magnitude
        pg = get_param(simdir, "pg")
        all_L.append(L)
        all_W.append(W)
        all_v.append(v)
        all_pg.append(pg)

        print(f"pg = {pg:.4e}, l = {L:.4e} um, w = {W:.4e} um, v = {v:.4e} cm/s")


    if args.out is not None:
        df = pd.DataFrame({"pg" : all_pg,
                           "l": all_L,
                           "w": all_W,
                           "v": all_v})
        df.sort_values(by="pg", inplace=True)
        df.to_csv(args.out, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
