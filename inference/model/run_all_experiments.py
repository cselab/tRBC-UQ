#!/usr/bin/env python

from dataclasses import dataclass
import pint
import numpy as np
import sys
import trimesh

from mpi4py import MPI

from dpdprops import (JuelicherLimRBCDefaultParams,
                      equivalent_sphere_radius)

# very hacky but this will do for now
if __name__ == '__main__':
    from tools import center_align_mesh
    from stressfree_gen import generate_stressfree_mesh
    from equilibration import run_equilibration
    from stretching import run_stretching
    from relaxation import run_relaxation
else:
    from .tools import center_align_mesh
    from .stressfree_gen import generate_stressfree_mesh
    from .equilibration import run_equilibration
    from .stretching import run_stretching
    from .relaxation import run_relaxation

def load_mesh(name: str):
    return trimesh.load(name, process=False)


@dataclass
class ExperimentsOutput:
    """
    Output of all experiments.
    """
    # from equilibrium experiment:
    eq_D: pint.Quantity
    eq_hmin: pint.Quantity
    eq_hmax: pint.Quantity

    # from stretching experiment:
    stretch_Fext: list
    stretch_D0: list
    stretch_D1: list

    # from relaxation experiment (None if disabled)
    relax_tc: pint.Quantity=None


def run_experiments(*,
                    ureg: pint.UnitRegistry,
                    rbc_params,
                    reduced_volume: float,
                    Fext_: list,
                    mesh_sphere: str,
                    mesh_ini_eq: str,
                    comm_address,
                    run_eq: bool=True,
                    run_stretch: bool=True,
                    run_relax: bool=True,
                    verbose: bool=False,
                    dump: bool=False,
                    stressfree_fname: str=None,
                    equilibrated_fname: str=None,
                    stretched_fname: str=None):
    """
    Run experiments for a given set of parameters.

    Args:
        ureg: The unit registry used for all experiments. Must coincide with that of rbc_params.
        rbc_params: Set of physical RBC parameters. Also defines which model to use (see dpdprops)
        reduced_volume: Reduced volume of the stressfree shape S0.
        Fext_: list of stretching forces to run.
        mesh_sphere: The mesh of a sphere, with the same faces as the final RBC mesh.
        mesh_ini_eq: The mesh of the starting point for equilibration.
        comm_address: Address of the MPI communicator.
        run_eq: Wether to run the equilibrium experiment or not. Must be set to True if run_stretch is set.
        run_stretch: Wether to run the stretching experiment or not. Must be set to True if run_relax is set.
        run_relax: Wether to run the relaxation experiment or not.
        verbose: Turn on the verbosity
        dump: dump the ply files for each experiment.
        stressfree_fname: if set, the name of the stressfree mesh file to be dumped
        equilibrated_fname: if set, the name of the equlibrated mesh file to be dumped
        stretched_fname: if set, the name of the stretched mesh file to be dumped

    Return:
        is_master: boolean, True if the current rank is the main rank.
        experiments_output: ExperimentsOutput object for the given parameters. None if run_eq is False.
    """

    # NOTE
    # here only the first rank needs the mesh, so we do not do anything in particular.
    # if the other ranks needed it, one would need either to synchronize an read from disk
    # ok send it via MPI. In both cases that would mean that one need a communicator
    # (instead of the address of a comminicator) but this is not supported by korali yet.

    if run_relax and not run_stretch:
        raise ValueError("Must set 'run_stretch' to True when 'run_relax' is set.")

    if run_stretch and not run_eq:
        raise ValueError("Must set 'run_eq' to True when 'run_stretch' is set.")


    mesh_ini = load_mesh(mesh_sphere)
    mesh_ref = load_mesh(mesh_sphere)

    rV = reduced_volume

    is_master, stressfree_mesh = generate_stressfree_mesh(ureg=ureg,
                                                          comm_address=comm_address,
                                                          reduced_volume=rV,
                                                          dump=dump,
                                                          mesh_ini=mesh_ini,
                                                          mesh_ref=mesh_ref)

    if is_master:
        stressfree_mesh = center_align_mesh(stressfree_mesh)
        if stressfree_fname is not None:
            stressfree_mesh.export(stressfree_fname)
        if verbose:
            print("Done stressfree shape generation.")
            sys.stdout.flush()

    if not run_eq:
        return is_master, None

    mesh_ini = load_mesh(mesh_ini_eq)

    is_master, equilibrated_mesh, D, hmin, hmax = run_equilibration(ureg=ureg,
                                                                    comm_address=comm_address,
                                                                    mesh_ini=mesh_ini,
                                                                    mesh_ref=stressfree_mesh,
                                                                    params=rbc_params,
                                                                    dump=dump)

    if is_master:
        equilibrated_mesh = center_align_mesh(equilibrated_mesh)
        if equilibrated_fname is not None:
            equilibrated_mesh.export(equilibrated_fname)

        if verbose:
            print("Done equilibration.")
            sys.stdout.flush()


    if not run_stretch:
        return is_master, ExperimentsOutput(eq_D = D,
                                            eq_hmin = hmin,
                                            eq_hmax = hmax,
                                            stretch_Fext = Fext_,
                                            stretch_D0 = [],
                                            stretch_D1 = [],
                                            relax_tc = None)


    # get stretching output quantities of interest for the given force
    D0_ = []
    D1_ = []

    for force_input_ in Fext_:
        is_master, _, Dshort, Dlong = run_stretching(ureg=ureg,
                                                     comm_address=comm_address,
                                                     mesh_ini=equilibrated_mesh,
                                                     mesh_ref=stressfree_mesh,
                                                     params=rbc_params,
                                                     dump=dump,
                                                     force_=force_input_)
        D0_.append(Dshort)
        D1_.append(Dlong)

    if is_master and verbose:
        print("Done stretching.")
        sys.stdout.flush()


    tc_ = None

    if run_relax:
        # get the stretched mesh before relaxing

        RA_ = equivalent_sphere_radius(area=rbc_params.A0)
        dimless_force = 2.3 # F / (RA * mu)
        force_before_relax_ = (rbc_params.mu * RA_ * dimless_force).to(ureg.pN)

        is_master, stretched_mesh, _, _ = run_stretching(ureg=ureg,
                                                         comm_address=comm_address,
                                                         mesh_ini=equilibrated_mesh,
                                                         mesh_ref=stressfree_mesh,
                                                         params=rbc_params,
                                                         dump=dump,
                                                         force_=force_before_relax_)


        if is_master:
            stretched_mesh = center_align_mesh(stretched_mesh)
            if stretched_fname is not None:
                stretched_mesh.export(stretched_fname)


        is_master, tc_ = run_relaxation(ureg=ureg,
                                        comm_address=comm_address,
                                        mesh_ini=stretched_mesh,
                                        mesh_ref=stressfree_mesh,
                                        params=rbc_params,
                                        dump=dump)

        if is_master and verbose:
            print("Done relaxation.")
            sys.stdout.flush()

    return is_master, ExperimentsOutput(eq_D = D,
                                        eq_hmin = hmin,
                                        eq_hmax = hmax,
                                        stretch_Fext = Fext_,
                                        stretch_D0 = D0_,
                                        stretch_D1 = D1_,
                                        relax_tc = tc_)


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Run all experiments for a set of parameters')
    parser.add_argument('--mesh-sphere', type=str, required=True, help="Initial spherical mesh.")
    parser.add_argument('--mesh-ini-eq', type=str, required=True, help="Initial mesh to use for equilibration step.")
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    comm_address = MPI._addressof(comm)

    ureg = pint.UnitRegistry()

    v, mu, FvK, b2, etam, Fext = 0.95, 4.0, 250.00000000000003, 1.6, 0.7, 100.0

    RA = np.sqrt(135/(4 * np.pi)) * ureg.um

    ka = mu

    mu *= ureg.uN / ureg.m
    ka *= ureg.uN / ureg.m
    kb = (mu * RA**2 / FvK).to(ureg.J)
    etam *= ureg.Pa * ureg.s * ureg.um
    Fext *= ureg.pN

    rbc_params = JuelicherLimRBCDefaultParams(ureg, mu=mu, ka=ka, kappab=kb, b2=b2, eta_m=etam)

    is_master, res = run_experiments(ureg=ureg,
                                     reduced_volume=v,
                                     rbc_params=rbc_params,
                                     Fext_=[Fext],
                                     mesh_sphere=args.mesh_sphere,
                                     mesh_ini_eq=args.mesh_ini_eq,
                                     comm_address=comm_address,
                                     run_stretch=True,
                                     run_relax=True,
                                     verbose=True,
                                     dump=True,
                                     stressfree_fname='S0.off',
                                     equilibrated_fname='eq.off',
                                     stretched_fname="stretched.off")

    if is_master:
        print(res)



if __name__ == '__main__':
    main(sys.argv[1:])
