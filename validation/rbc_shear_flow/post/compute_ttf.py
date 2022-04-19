#!/usr/bin/env python

desc = """
Compute dimensionless the tank treading frequency (TTF) 4 pi f / gamma_dot,
where gamma_dot is the shear rate and f is the TTF.
"""

import argparse
import glob
import numpy as np
import os
import sys
import trimesh

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from parameters import (RBCShearParams, load_parameters)

def collect_xpositions(mesh_files: list):
    """
    Collect the x coordinates of all vertices of a sequence of mesh.
    Arguments:
        mesh_files: list of mesh files (str)
    Return:
        the x positions of all mesh.
    """
    num_mesh = len(mesh_files)
    all_x = []

    for f in mesh_files:
        mesh = trimesh.load(f, process=False)
        pos = mesh.vertices - mesh.center_mass
        x = pos[:,0]
        all_x.append(x)

    return np.array(all_x)


def compute_period(t: np.ndarray,
                   x: np.ndarray):
    """
    Compute the period of a time sequence x(t).
    """
    start = len(x) // 4
    t = t[start:]
    x = x[start:]

    zero_crossings = np.where(np.diff(np.sign(x)))[0]

    if len(zero_crossings) < 2:
        raise RuntimeError(f"found only {len(zero_crossings)} zero-crossing. Cannot compute the TTF. Run the simulation for a longer time.")

    tcross = t[zero_crossings]
    tinter = np.diff(tcross)
    return 2 * np.mean(tinter) # 2 because crosses twice zero in a period.


def compute_dimensionless_ttf(basedir: str):
    """
    Compute the tank-treading frequency (ttf) of a given case of RBC under shear flow, in units of shear rate / 4pi.

    Arguments:
        basedir: the base directory of the simulation
    Returns:
        The TTF in units of shear_rate.
    """
    p = load_parameters(os.path.join(basedir, "parameters.pkl"))

    mesh_files = sorted(glob.glob(os.path.join(basedir, 'ply', '*.ply')))
    all_x = collect_xpositions(mesh_files)
    num_times, num_vertices = all_x.shape

    stds = np.std(all_x, axis=1)
    std_threshold = np.quantile(stds, q=0.9)
    idx = np.argwhere(stds >= std_threshold).flatten()
    all_x = all_x[:,idx]

    t = np.arange(num_times) * p.t_dump_every
    periods = []

    for x in all_x.T:
        try:
            T = compute_period(t,x)
            periods.append(T)
        except:
            pass
    T = np.mean(periods)
    ttf = 1/T

    return 4 * np.pi * ttf / p.shear_rate

def compute_dimensionless_ttf_fft(basedir: str):
    """
    Compute the tank-treading frequency (ttf) of a given case of RBC under shear flow, in units of shear rate / 4pi, using fft.

    Arguments:
        basedir: the base directory of the simulation
    Returns:
        The TTF in units of shear_rate.
    """
    p = load_parameters(os.path.join(basedir, "parameters.pkl"))

    mesh_files = sorted(glob.glob(os.path.join(basedir, 'ply', '*.ply')))
    all_x = collect_xpositions(mesh_files)
    num_times, num_vertices = all_x.shape
    t = np.arange(num_times) * p.t_dump_every

    # select the points with the largest amplitude
    stds = np.std(all_x, axis=1)
    std_threshold = np.quantile(stds, q=0.9)
    idx = np.argwhere(stds >= std_threshold).flatten()
    all_x = all_x[:,idx]

    # center
    all_x -= np.mean(all_x, axis=0)[np.newaxis,:]

    all_q = np.fft.fft(all_x, axis=0)
    nu = np.fft.fftfreq(num_times, d=p.t_dump_every)

    idx = np.argmax(np.abs(all_q), axis=0)
    ttf = np.mean(np.abs(nu[idx]))

    return 4 * np.pi * ttf / p.shear_rate


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('basedir', type=str, help="The directory containing the simulation results.")
    args = parser.parse_args()

    ttf = compute_dimensionless_ttf(args.basedir)
    print(f"estimate from period: {ttf}")
    ttf = compute_dimensionless_ttf_fft(args.basedir)
    print(f"estimate from fft: {ttf}")
