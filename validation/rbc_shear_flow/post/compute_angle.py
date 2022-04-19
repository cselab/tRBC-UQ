#!/usr/bin/env python

desc = """
Compute the inclination angle of a red blood cell in shear.
"""

import argparse
import glob
import numpy as np
import os
import sys
import trimesh
from scipy.optimize import least_squares

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from parameters import (RBCShearParams, load_parameters)


def compute_inclination_angle(mesh_file: str):
    mesh = trimesh.load(mesh_file, process=False)
    pos = mesh.vertices - mesh.center_mass
    x = pos[:,0]
    y = pos[:,1]
    data = np.zeros((len(x), 2))
    data[:,0] = x
    data[:,1] = y

    eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
    direction = eigenvectors[0] # stored in decreasing order of corresponding eigen values
    if direction[0] < 0:
        direction = -direction

    theta = np.arctan2(direction[1], direction[0])
    return theta


def compute_mean_inclination_angle(basedir: str,
                                   plot: bool=False):
    """
    Compute the inclination angle of a given case of RBC under shear flow, in radians

    Arguments:
        basedir: the base directory of the simulation
        plot: if True, show a plot of angle against dump iteration.
    Returns:
        The angle of inclination
    """

    p = load_parameters(os.path.join(basedir, "parameters.pkl"))
    mesh_files = sorted(glob.glob(os.path.join(basedir, 'ply', '*.ply')))
    start = len(mesh_files) // 4

    thetas = []
    for f in mesh_files[start:]:
        theta = compute_inclination_angle(f)
        thetas.append(theta)

    thetas = np.array(thetas)
    t = np.arange(len(thetas)) * p.t_dump_every

    def model(t, a, b, nu, phi):
        return a + b * np.cos(t * nu * 2 * np.pi + phi)

    def f(x):
        a, b, nu, phi = x
        thetas_th = model(t, a, b, nu, phi)
        return thetas - thetas_th

    x0 = [np.mean(thetas), np.ptp(thetas)/2, p.shear_rate/(4*np.pi), 0]
    res = least_squares(f, x0=x0)

    a, b, nu, phi = res.x

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(t, thetas * 180 / np.pi)
        ax.plot(t, model(t, a, b, nu, phi)  * 180/np.pi, '--k')
        ax.set_xlabel('time')
        ax.set_ylabel('angle (degrees)')
        plt.show()

    return a, b


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('basedir', type=str, help="The directory containing the simulation results.")
    parser.add_argument('--plot', action='store_true', default=False, help="If set, plot the angle over time for the given simulation.")
    args = parser.parse_args()

    theta, dtheta = compute_mean_inclination_angle(args.basedir, args.plot)
    print(theta * 180 / np.pi)
