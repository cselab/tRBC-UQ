#!/usr/bin/env python

import mirheo as mir
import numpy as np
import sys

from parameters import (PipeFlowParams,
                        load_parameters)
import objplacement


def generate_cells(p: 'PipeFlowParams',
                   *,
                   scale_ini: float,
                   ranks: tuple=(1,1,1)):

    m  = p.m
    rc = p.rc
    L  = p.L
    R  = p.R
    RA = p.RA

    drag = 4

    domain = (L, 2*R + 4*rc, 2*R + 4*rc)
    # com_q = [[domain[0]/4, domain[1]/2, domain[2]/2,
    #           np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]]
    com_q = [[domain[0]/4, domain[1]/2, domain[2]/2,
              1, 0, 0, 0]]

    T = np.sqrt(m * RA**2  / p.rbc_params.bending_modulus())
    tend = 50 * T
    dt = 0.001

    niters = int(tend/dt)

    checkpoint_every = niters-5

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='generate_ic',
                   checkpoint_folder="generate_ic/", checkpoint_every=checkpoint_every)

    max_contact_force = p.max_contact_force

    p.rbc_params.gamma = 300
    rbc_int = mir.Interactions.MembraneForces('int_rbc', **p.rbc_params.to_interactions(), stress_free=True,
                                              grow_until = tend*0.5, init_length_fraction=scale_ini)
    u.registerInteraction(rbc_int)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)

    wall = mir.Walls.Cylinder('pipe', center=(domain[1]/2, domain[2]/2),
                              radius=R, axis='x', inside=True)
    u.registerWall(wall)

    mesh_rbc = mir.ParticleVectors.MembraneMesh(p.mesh_ini.vertices.tolist(),
                                                p.mesh_ref.vertices.tolist(),
                                                p.mesh_ini.faces.tolist())
    pv_rbc = mir.ParticleVectors.MembraneVector('rbc', mass=m, mesh=mesh_rbc)
    u.registerParticleVector(pv_rbc, mir.InitialConditions.Membrane(com_q, global_scale=scale_ini))

    u.setInteraction(rbc_int, pv_rbc, pv_rbc)
    u.setIntegrator(vv, pv_rbc)

    h = 0.1 * rc
    u.registerPlugins(mir.Plugins.createWallRepulsion("wall_force", pv_rbc, wall, C=max_contact_force/h, h=h, max_force=max_contact_force))
    u.registerPlugins(mir.Plugins.createParticleDrag("drag", pv_rbc, drag))

    if u.isMasterTask():
        print(f"domain={domain}")
        print(f"L={L}, R={R}, RA={RA}")
        sys.stdout.flush()

    u.dumpWalls2XDMF([wall], h=(1,1,1), filename='h5/wall')

    dump_every = int(10.0/dt)
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, 'ply_generate_ic/'))
    u.registerPlugins(mir.Plugins.createStats('stats', every=dump_every))

    u.run(niters, dt)


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Generate cells with given hematocrit in pipe.')
    parser.add_argument('--params', type=str, default="parameters.pkl", help="Parameters file.")
    parser.add_argument('--ranks', type=int, nargs=3, default=(1, 1, 1), help="Ranks in each dimension.")
    parser.add_argument('--scale-ini', type=float, default=0.5, help="Initial size of the RBCs.")
    args = parser.parse_args(argv)

    p = load_parameters(args.params)

    generate_cells(p,
                   scale_ini=args.scale_ini,
                   ranks=tuple(args.ranks))

if __name__ == '__main__':
    main(sys.argv[1:])
