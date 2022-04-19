#!/usr/bin/env python

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import trimesh

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, ".."))

from parameters import (PipeFlowParams,
                        load_parameters)

def load_mesh(fname: str):
    return trimesh.load(fname, process=False)

def compute_area_volumes(simdir: str):
    ply_list = sorted(glob.glob(os.path.join(simdir, 'ply', 'rbc_*.ply')))
    params_path = os.path.join(simdir, 'settings.pkl')
    params = load_parameters(params_path)

    areas = []
    volumes = []

    for fname in ply_list:
        mesh = load_mesh(fname)
        areas.append(mesh.area)
        volumes.append(mesh.volume)

    areas = np.array(areas)
    volumes = np.array(volumes)

    A0 = params.rbc_params.area
    V0 = params.rbc_params.volume

    dA = (areas - A0) / A0
    dV = (volumes - V0) / V0

    n = len(ply_list)

    fig, ax = plt.subplots()
    ax.plot(range(n), dA, label='area')
    ax.plot(range(n), dV, label='volume')
    ax.set_xlabel("dump id")
    ax.set_ylabel("relative error")
    ax.legend()
    plt.show()


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Compute the area and volume of the RBC flowing in the pipe over time.')
    parser.add_argument('simdir', type=str, help="The simulation directory.")
    args = parser.parse_args(argv)

    compute_area_volumes(args.simdir)


if __name__ == '__main__':
    main(sys.argv[1:])
