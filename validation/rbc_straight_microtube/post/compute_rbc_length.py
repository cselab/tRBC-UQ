#!/usr/bin/env python

import glob
import numpy as np
import os
import pandas as pd
import sys
import trimesh

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, ".."))

from parameters import (PipeFlowParams,
                        load_parameters)

def load_mesh(fname: str):
    return trimesh.load(fname, process=False)

def compute_mean_rbc_length_width(simdir: str,
                                  show_plot: bool=False):
    ply_list = sorted(glob.glob(os.path.join(simdir, 'ply', 'rbc_*.ply')))
    params_path = os.path.join(simdir, 'settings.pkl')
    params = load_parameters(params_path)

    n = len(ply_list)
    start = 0
    lengths = []
    widths = []

    for fname in ply_list[start:]:
        mesh = load_mesh(fname)
        L = np.ptp(mesh.vertices[:,0]) * params.length_scale_
        W = np.ptp(mesh.vertices[:,1]) * params.length_scale_
        lengths += [L.to('um').magnitude]
        widths  += [W.to('um').magnitude]

    lengths = np.array(lengths)
    widths = np.array(widths)
    n = len(lengths)

    if show_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(range(n), lengths)
        ax.set_xlabel("dump id")
        ax.set_ylabel(r"$l [\mu m]$")
        plt.show()

    start = 7*n//8
    return np.mean(lengths[n//2:]), np.mean(widths[n//2:])

def get_param(basedir: str, param_name: str):
    import re
    rexf = '[-+]?\d*\.\d+|\d+'
    matches = re.findall(f"{param_name}_({rexf})", basedir)
    assert len(matches) == 1
    return float(matches[0])

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Compute the mean length and width of the RBC flowing in the pipe.')
    parser.add_argument('simdirs', type=str, nargs='+', help="The simulation directories.")
    parser.add_argument('--out', type=str, default=None, help="output csv file.")
    parser.add_argument('--against', type=str, default="pg", help="Varying input parameter.")
    parser.add_argument('--show-plot', action='store_true', default=False, help="If true, show the RBC length over time.")
    args = parser.parse_args(argv)

    param_name = args.against
    all_L = list()
    all_W = list()
    all_against_params = list()

    for simdir in args.simdirs:
        L, W = compute_mean_rbc_length_width(simdir,
                                             show_plot=args.show_plot)
        param_value = get_param(simdir, param_name)
        print(f"{param_name} = {param_value:.4e}, l = {L:.4e} um, w = {W:.4e} um")
        all_L.append(L)
        all_W.append(W)
        all_against_params.append(param_value)

    if args.out is not None:
        df = pd.DataFrame({param_name : all_against_params,
                           "l": all_L,
                           "w": all_W})
        df.to_csv(args.out, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
