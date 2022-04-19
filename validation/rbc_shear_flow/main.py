#!/usr/bin/env python

from copy import deepcopy
import mirheo as mir
from mpi4py import MPI
import numpy as np
import pint
import sys
import trimesh

import dpdprops
from parameters import RBCShearParams, load_parameters

def run_shear_flow(*,
                   p: RBCShearParams,
                   comm: MPI.Comm,
                   ranks: tuple=(1,1,1),
                   dump: bool=False):
    """
    Argument:
        p: The parameters of the simulation
        dump: if True, will dump the mesh over time.
    """
    m = p.m
    nd = p.nd
    rc = p.rc
    L = p.L
    shear_rate = p.shear_rate

    domain = (L, L + 4*rc, L)

    u = mir.Mirheo(ranks, domain, debug_level=0, log_filename='log', no_splash=True, comm_ptr=MPI._addressof(comm))


    # Membrane

    mesh_rbc = mir.ParticleVectors.MembraneMesh(p.mesh_ini.vertices.tolist(),
                                                p.mesh_ref.vertices.tolist(),
                                                p.mesh_ini.faces.tolist())
    pv_rbc = mir.ParticleVectors.MembraneVector("pv_rbc", mass=m, mesh=mesh_rbc)
    ic_rbc = mir.InitialConditions.Membrane([[domain[0] * 0.5,
                                              domain[1] * 0.5,
                                              domain[2] * 0.5,
                                              1.0, 0.0, 0.0, 0.0]])
    u.registerParticleVector(pv_rbc, ic_rbc)


    # Solvent

    pv_outer = mir.ParticleVectors.ParticleVector('pv_outer', mass=m)
    ic_outer = mir.InitialConditions.Uniform(number_density=nd)
    u.registerParticleVector(pv=pv_outer, ic=ic_outer)

    rbc_checker = mir.BelongingCheckers.Mesh('rbc_checker')
    u.registerObjectBelongingChecker(rbc_checker, pv_rbc)
    pv_inner = u.applyObjectBelongingChecker(rbc_checker, pv_outer, inside='pv_inner', correct_every=10000)

    # Interactions

    dpd_oo = mir.Interactions.Pairwise('dpd_oo', rc=rc, kind="DPD", **p.dpd_oo.to_interactions())
    dpd_ii = mir.Interactions.Pairwise('dpd_ii', rc=rc, kind="DPD", **p.dpd_ii.to_interactions())
    dpd_io = mir.Interactions.Pairwise('dpd_io', rc=rc, kind="DPD", **p.dpd_io.to_interactions())
    dpd_rbco = mir.Interactions.Pairwise('dpd_rbco', rc=rc, kind="DPD", **p.dpd_rbco.to_interactions())
    dpd_rbci = mir.Interactions.Pairwise('dpd_rbci', rc=rc, kind="DPD", **p.dpd_rbci.to_interactions())

    u.registerInteraction(dpd_oo)
    u.registerInteraction(dpd_ii)
    u.registerInteraction(dpd_io)
    u.registerInteraction(dpd_rbco)
    u.registerInteraction(dpd_rbci)

    int_rbc = mir.Interactions.MembraneForces("int_rbc", **p.rbc_params.to_interactions(), stress_free=True)


    # integrators

    dt_solv = min(prms.get_max_dt() for prms in [p.dpd_oo, p.dpd_ii])
    dt_memb = p.rbc_params.get_max_dt(mass=m)

    substeps = int(dt_solv / dt_memb) + 1
    dt = min([substeps * dt_memb, dt_solv]) / 2

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)

    subvv = mir.Integrators.SubStep('subvv', substeps, [int_rbc])
    u.registerIntegrator(subvv)

    # Walls

    vw = shear_rate * L / 2

    plate_lo = mir.Walls.MovingPlane("plate_lo", normal=(0, -1, 0), pointThrough=(0,     2 * rc, 0), velocity=(-vw, 0, 0))
    plate_hi = mir.Walls.MovingPlane("plate_hi", normal=(0,  1, 0), pointThrough=(0, L + 2 * rc, 0), velocity=( vw, 0, 0))

    u.registerWall(plate_lo)
    u.registerWall(plate_hi)

    nequil = int(1.0 / dt)
    frozen_lo = u.makeFrozenWallParticles("plate_lo", walls=[plate_lo], interactions=[dpd_oo], integrator=vv, number_density=nd, nsteps=nequil, dt=dt)
    frozen_hi = u.makeFrozenWallParticles("plate_hi", walls=[plate_hi], interactions=[dpd_oo], integrator=vv, number_density=nd, nsteps=nequil, dt=dt)

    move_lo = mir.Integrators.Translate('move_lo', velocity=(-vw, 0, 0))
    move_hi = mir.Integrators.Translate('move_hi', velocity=( vw, 0, 0))
    u.registerIntegrator(move_lo)
    u.registerIntegrator(move_hi)


    # Set interactions between pvs

    u.setInteraction(dpd_oo, pv_outer, pv_outer)
    u.setInteraction(dpd_ii, pv_inner, pv_inner)
    u.setInteraction(dpd_io, pv_inner, pv_outer)

    u.setInteraction(dpd_rbco, pv_rbc, pv_outer)
    u.setInteraction(dpd_rbci, pv_rbc, pv_inner)

    u.setInteraction(dpd_oo, pv_outer, frozen_lo)
    u.setInteraction(dpd_oo, pv_outer, frozen_hi)

    # Note. we assume thet the membrane is always far from the wall, hence we do not add any
    # interactions between membrane - walls and pv_inner - walls.

    # Set integrators
    u.setIntegrator(vv, pv_outer)
    u.setIntegrator(vv, pv_inner)
    u.setIntegrator(subvv, pv_rbc)
    u.setIntegrator(move_lo, frozen_lo)
    u.setIntegrator(move_hi, frozen_hi)

    # Set Bouncers

    rbc_bouncer = mir.Bouncers.Mesh("rbc_bouncer", "bounce_maxwell", kBT=p.kBT)
    u.registerBouncer(rbc_bouncer)

    u.setBouncer(rbc_bouncer, pv_rbc, pv_outer)
    u.setBouncer(rbc_bouncer, pv_rbc, pv_inner)

    u.setWall(plate_lo, pv_outer)
    u.setWall(plate_hi, pv_outer)

    # Dump plugins
    if dump:
        t_dump_every = p.t_dump_every
        dump_every = int(t_dump_every / dt)
        path = 'ply/'
        u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, path))
        u.registerPlugins(mir.Plugins.createStats('stats', every=dump_every, filename="stats.csv"))


    omega_sphere = 0.5 * p.shear_rate
    f_sphere = omega_sphere / (2 * np.pi)
    # tend so that a sphere would make a given number of revolutions
    tend = 10 / f_sphere
    nsteps = int(tend / dt)

    if u.isMasterTask():
        print(f"Domain = {domain}")
        print(f"dt = {dt} ({substeps} substeps)")
        if dump:
            print(f"t_dump_every = {t_dump_every}")
        print(f"will run for {nsteps} steps ({tend} simulation time units)")
        sys.stdout.flush()

    u.run(nsteps, dt)



def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters', type=str, help="The file containing the parameters of the simulation.")
    parser.add_argument('--dump', action='store_true', default=False, help="Will dump ply files if set to True.")
    parser.add_argument('--ranks', type=int, nargs=3, default=[1,1,1], help="Number of ranks in each direction.")
    args = parser.parse_args(argv)

    p = load_parameters(args.parameters)

    comm = MPI.COMM_WORLD

    run_shear_flow(p=p,
                   ranks=args.ranks,
                   comm=comm,
                   dump=args.dump)


if __name__ == '__main__':
    main(sys.argv[1:])
