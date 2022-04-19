#!/usr/bin/env python

import mirheo as mir
import numpy as np
import sys

from parameters import (PipeFlowParams,
                        load_parameters)

def get_pv_num_particles(pv):
    return len(pv.getCoordinates())

def run_capillary_flow(p: 'PipeFlowParams',
                       *,
                       no_visc: bool,
                       ranks: tuple=(1,1,1),
                       restart_directory: str=None,
                       checkpoint_directory: str=None):
    """
    Arguments:
        p: the simulation parameters (see parameters.py).
        no_visc: if True, disable the viscosity of membrane.
        ranks: number of ranks per dimension.
        restart_directory: if set, specify from which directory to restart.
        checkpoint_directory: if set, specify to which directory to dump checkpoints.
    """

    rc   = p.rc
    L    = p.L
    R    = p.R
    Vmax = p.Vmax
    RA   = p.RA

    dt_solv = min([dpd.get_max_dt() for dpd in [p.dpd_ii, p.dpd_oo, p.dpd_rbco, p.dpd_rbci]])

    if no_visc:
        p.rbc_params.set_viscosity(0)

    tend = 5000 * R / Vmax

    domain = (L, 2*R + 4*rc, 2*R + 4*rc)

    if checkpoint_directory is not None:
        u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', checkpoint_every=10000, checkpoint_folder=checkpoint_directory)
    else:
        u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log')

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)

    # Membranes

    mesh_rbc = mir.ParticleVectors.MembraneMesh(p.mesh_ini.vertices.tolist(),
                                                p.mesh_ref.vertices.tolist(),
                                                p.mesh_ini.faces.tolist())
    pv_rbc = mir.ParticleVectors.MembraneVector('rbc', mass=p.m, mesh=mesh_rbc)
    u.registerParticleVector(pv_rbc, mir.InitialConditions.Restart('generate_ic'))

    # Solvent

    pv_outer = mir.ParticleVectors.ParticleVector('outer', mass=p.m)
    ic_outer = mir.InitialConditions.Uniform(number_density=p.nd)
    u.registerParticleVector(pv=pv_outer, ic=ic_outer)

    rbc_checker = mir.BelongingCheckers.Mesh('rbc_checker')
    u.registerObjectBelongingChecker(rbc_checker, pv_rbc)
    pv_inner = u.applyObjectBelongingChecker(rbc_checker, pv_outer, inside='inner', correct_every=10000)

    rbc_bouncer = mir.Bouncers.Mesh("rbc_bouncer", "bounce_maxwell", kBT=p.kBT)
    u.registerBouncer(rbc_bouncer)

    # Interactions

    dpd_oo = mir.Interactions.Pairwise('dpd_oo', rc=rc, kind="DPD", **p.dpd_oo.to_interactions())
    dpd_ii = mir.Interactions.Pairwise('dpd_ii', rc=rc, kind="DPD", **p.dpd_ii.to_interactions())
    dpd_io = mir.Interactions.Pairwise('dpd_io', rc=rc, kind="DPD", **p.dpd_io.to_interactions())

    u.registerInteraction(dpd_oo)
    u.registerInteraction(dpd_ii)
    u.registerInteraction(dpd_io)

    dpd_rbco = mir.Interactions.Pairwise('dpd_rbco', rc=rc, kind="DPD", **p.dpd_rbco.to_interactions())
    dpd_rbci = mir.Interactions.Pairwise('dpd_rbci', rc=rc, kind="DPD", **p.dpd_rbci.to_interactions())

    # Walls

    wall = mir.Walls.Cylinder('pipe', center=(domain[1]/2, domain[2]/2),
                              radius=R, axis='x', inside=True)
    u.registerWall(wall)
    frozen = u.makeFrozenWallParticles("frozen", walls=[wall], interactions=[dpd_oo], integrator=vv, number_density=p.nd, nsteps=int(1.0/dt_solv), dt=dt_solv)
    u.setWall(wall, pv_outer)

    # Set interactions between pvs

    u.registerInteraction(dpd_rbco)
    u.registerInteraction(dpd_rbci)

    u.setInteraction(dpd_oo, pv_outer, pv_outer)
    u.setInteraction(dpd_ii, pv_inner, pv_inner)
    u.setInteraction(dpd_io, pv_inner, pv_outer)

    u.setInteraction(dpd_rbco, pv_rbc, pv_outer)
    u.setInteraction(dpd_rbci, pv_rbc, pv_inner)

    u.setInteraction(dpd_oo, pv_outer, frozen)
    u.setInteraction(dpd_io, pv_inner, frozen)
    u.setInteraction(dpd_rbco, pv_rbc, frozen)

    # set bouncers
    u.setBouncer(rbc_bouncer, pv_rbc, pv_inner)
    u.setBouncer(rbc_bouncer, pv_rbc, pv_outer)

    # Integrators

    u.setIntegrator(vv, pv_outer)
    u.setIntegrator(vv, pv_inner)

    dt_rbc_el = p.rbc_params.get_max_dt_elastic(mass=p.m)
    substeps = 1 + int(dt_solv / dt_rbc_el)
    dt = dt_solv

    if no_visc:
        rbc_int = mir.Interactions.MembraneForces('int_rbc', **p.rbc_params.to_interactions(), stress_free=True)
        u.registerInteraction(rbc_int)

        if substeps == 1:
            u.setIntegrator(vv, pv_rbc)
            u.setInteraction(rbc_int, pv_rbc, pv_rbc)
        else:
            vv_rbc = mir.Integrators.SubStep("vv_rbc", substeps, fastForces=[rbc_int])
            u.registerIntegrator(vv_rbc)
            u.setIntegrator(vv_rbc, pv_rbc)

    else: # we use Shardlow integration
        dt_rbc_visc = p.rbc_params.get_max_dt_visc(mass=p.m)

        rbc_int = mir.Interactions.MembraneForces('int_rbc', **p.rbc_params.to_interactions_zero_visc(), stress_free=True)
        u.registerInteraction(rbc_int)

        vv_rbc = mir.Integrators.SubStepShardlowSweep("vv_rbc", substeps=10, fastForces=rbc_int, nsweeps=20, **p.rbc_params.to_viscous())
        u.registerIntegrator(vv_rbc)
        u.setIntegrator(vv_rbc, pv_rbc)


    # Plugins

    t_dump_every = 0.2 * R / Vmax
    dump_every = int(t_dump_every/dt)
    stats_every = dump_every

    h = 0.1 * rc
    u.registerPlugins(mir.Plugins.createWallRepulsion("wall_force", pv_rbc, wall, C=p.max_contact_force/h, h=h, max_force=p.max_contact_force))

    # estimate the number density of particles, accounting for pv_rbcs
    rho = p.nd

    f = (p.pressure_gradient / p.nd, 0, 0)
    u.registerPlugins(mir.Plugins.createAddForce("body_force_solvent", pv_outer, f))
    u.registerPlugins(mir.Plugins.createAddForce("body_force_hemoglobine", pv_inner, f))

    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, 'ply/'))
    u.registerPlugins(mir.Plugins.createStats("stats", every=stats_every, filename='stats.csv', pvs=[pv_inner, pv_outer, pv_rbc]))
    u.registerPlugins(mir.Plugins.createDumpObjectStats("rbc_stats", ov=pv_rbc, dump_every=stats_every, path="obj_stats"))

    if u.isMasterTask():
        print(f"tend = {tend}")
        print(f"dt = {dt}")
        print(f"substeps = {substeps}")
        sys.stdout.flush()

    u.dumpWalls2XDMF([wall], h=(1,1,1), filename='h5/wall')

    if restart_directory is not None:
        u.restart(restart_directory)

    u.run(int(tend/dt), dt)


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Run RBCs flowing in a capillary pipe.')
    parser.add_argument('--params', type=str, default="parameters.pkl", help="Parameters file.")
    parser.add_argument('--ranks', type=int, nargs=3, default=(1, 1, 1), help="Ranks in each dimension.")
    parser.add_argument('--no-visc', action='store_true', default=False, help="Set zero membrane viscosity.")
    parser.add_argument('--restart-dir', type=str, default=None, help="The restart directory name (no restart if not set)")
    parser.add_argument('--checkpoint-dir', type=str, default=None, help="The checkpoint directory name (no checkpoint if not set)")
    args = parser.parse_args(argv)

    p = load_parameters(args.params)

    run_capillary_flow(p,
                       no_visc = args.no_visc,
                       ranks=tuple(args.ranks),
                       restart_directory=args.restart_dir,
                       checkpoint_directory=args.checkpoint_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
