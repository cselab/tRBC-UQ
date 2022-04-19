#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import sys

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, ".."))

from parameters import (PipeFlowParams,
                        load_parameters)

def unroll_periodic(x: np.ndarray,
                    L: float):
    dx = np.diff(x)
    jumps = np.zeros_like(dx)
    jumps -= L * (dx > +L/2)
    jumps += L * (dx < -L/2)
    x[1:] += np.cumsum(jumps)
    return x

def compute_mean_rbc_velocity(simdir: str,
                              show_plot: bool=False):
    rbc_csv_path = os.path.join(simdir, 'obj_stats', 'rbc.csv')
    params_path = os.path.join(simdir, 'settings.pkl')
    params = load_parameters(params_path)

    df = pd.read_csv(rbc_csv_path)
    t = df['time'].to_numpy()
    x = df['comx'].to_numpy()
    x = unroll_periodic(x, params.L)

    start = (7*len(x))//8
    pol = np.polyfit(t[start:], x[start:], deg=1)

    if show_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(t[:-1], np.diff(x)/np.diff(t))
        #ax.plot(t, np.polyval(pol,t), '--k')
        ax.set_xlabel(r"$t$ (a.u.)")
        ax.set_ylabel(r"$v$ (a.u.)")
        plt.show()

    v = pol[0] * params.length_scale_ / params.time_scale_ / params.shear_scale_factor
    return v

def get_param(basedir: str, param_name: str):
    import re
    rexf = '[-+]?\d*\.\d+|\d+'
    matches = re.findall(f"{param_name}_({rexf})", basedir)
    assert len(matches) == 1
    return float(matches[0])

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Compute the mean velocity of the RBC flowing in the pipe.')
    parser.add_argument('simdirs', type=str, nargs='+', help="The simulation directories.")
    parser.add_argument('--out', type=str, default=None, help="output csv file.")
    parser.add_argument('--against', type=str, default="pg", help="Varying input parameter.")
    parser.add_argument('--show-plot', action='store_true', default=False, help="If true, show the RBC velocity over time.")
    args = parser.parse_args(argv)

    param_name = args.against
    all_v = list()
    all_against_params = list()

    for simdir in args.simdirs:
        v = compute_mean_rbc_velocity(simdir, show_plot=args.show_plot)
        param_value = get_param(simdir, param_name)
        print(f"{param_name} = {param_value:.4e}, v = {v:.4e}")
        all_v.append(v.to('cm/s').magnitude)
        all_against_params.append(param_value)

    if args.out is not None:
        df = pd.DataFrame({param_name : all_against_params,
                           "v": all_v})
        df.to_csv(args.out, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
