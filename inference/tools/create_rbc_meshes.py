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
    parser = argparse.ArgumentParser(description='Create RBC stress-free shape and equilibrium shape meshes.')
    parser.add_argument('--reduced-volume', type=float, default=0.93, help="Reduced volume of the stress-free shape.")
    parser.add_argument('--mesh-sphere', type=str, required=True, help="Initial spherical mesh.")
    parser.add_argument('--out-sf-mesh', type=str, default=None, help="Output stress-free mesh.")
    parser.add_argument('--out-eq-mesh', type=str, default=None, help="Output equilibrated mesh.")
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    ureg = pint.UnitRegistry()

    rbc_params = JuelicherLimRBCDefaultParams(ureg)

    v_RBC = rbc_params.get_reduced_volume()

    sf_fname = args.out_sf_mesh
    eq_fname = args.out_eq_mesh

    sf_vrbc_fname = "S0_v_RBC.off"
    verbose=True
    dump=False

    # first create a cell that has the reduced volume of a healthy RBC.
    # this will be used as the initial condition for the equilibrium esperiment.
    run_experiments(ureg=ureg,
                    reduced_volume=v_RBC,
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
                    stressfree_fname=sf_vrbc_fname,
                    equilibrated_fname=None)


    # create the stress free shape with the required reduced volume, and then run equlibration.
    run_experiments(ureg=ureg,
                    reduced_volume=args.reduced_volume,
                    rbc_params=rbc_params,
                    Fext_=[],
                    mesh_sphere=args.mesh_sphere,
                    mesh_ini_eq=sf_vrbc_fname,
                    comm_address=MPI._addressof(comm),
                    run_stretch=False,
                    run_relax=False,
                    verbose=verbose,
                    dump=dump,
                    stressfree_fname=sf_fname,
                    equilibrated_fname=eq_fname)



if __name__ == '__main__':
    main(sys.argv[1:])
