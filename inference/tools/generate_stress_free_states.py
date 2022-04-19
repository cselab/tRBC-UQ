#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import os
import pint
import sys

from dpdprops import JuelicherLimRBCDefaultParams

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from model import run_experiments

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Create stress-free state meshes.')
    parser.add_argument('--mesh-sphere', type=str, required=True, help="Initial spherical mesh.")
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    ureg = pint.UnitRegistry()

    rbc_params = JuelicherLimRBCDefaultParams(ureg)

    for v in [0.65, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
        sf_fname = f"S0_v_{v}.ply"
        verbose=True
        dump=False

        print(f"Generating {sf_fname}")
        run_experiments(ureg=ureg,
                        reduced_volume=v,
                        rbc_params=rbc_params,
                        Fext_=[],
                        mesh_sphere=args.mesh_sphere,
                        mesh_ini_eq=None,
                        comm_address=MPI._addressof(comm),
                        run_eq=False,
                        run_stretch=False,
                        run_relax=False,
                        verbose=verbose,
                        dump=dump,
                        stressfree_fname=sf_fname,
                        equilibrated_fname=None)



if __name__ == '__main__':
    main(sys.argv[1:])
