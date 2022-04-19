#!/usr/bin/env python

import copy
import numpy as np
import os
import pandas as pd
import pint
import sys
import trimesh

import mirheo as mir
from dpdprops import JuelicherLimRBCDefaultParams

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from tools import (center_align_mesh,
                   compute_diameters,
                   compute_micro_beads_forces)


def rescale_by_area(mesh, A):
    mesh.vertices *= np.sqrt(A / mesh.area)
    return mesh

def run_stretching(*,
                   ureg,
                   comm_address,
                   mesh_ini,
                   mesh_ref,
                   force_: float,
                   params,
                   RA: float=1,
                   dump: bool=False):
    """
    Parameters:
        ureg: the unit registry
        comm_address: the address of the MPI communicator
        mesh_ini: Initial mesh
        mesh_ref: stress free mesh
        force_: The applied stretching force, in [pN]
        params: The RBC parameters (see dpdprops).
        RA: equivalent radius of the membrane in simulation units (sets the length scale)
        dump: if True, dump ply files.
    Returns:
        D0, D1: The diameters of the RBC, in [um]
    """

    assert force_.check('[force]')

    A0 = 4 * np.pi * RA**2
    mesh_ini = rescale_by_area(mesh_ini, A0)
    mesh_ref = rescale_by_area(mesh_ref, A0)

    ranks  = (1, 1, 1)

    safety = 3
    mass = 1

    domain = np.ptp(mesh_ini.vertices, axis=0) * safety
    domain = tuple(np.array(domain, dtype=int))

    u = mir.Mirheo(ranks, domain, debug_level=0, log_filename='log', no_splash=True, comm_ptr=comm_address)

    mesh_rbc = mir.ParticleVectors.MembraneMesh(mesh_ini.vertices.tolist(),
                                                mesh_ref.vertices.tolist(),
                                                mesh_ini.faces.tolist())
    pv_rbc = mir.ParticleVectors.MembraneVector("rbc", mass=mass, mesh=mesh_rbc)
    ic_rbc = mir.InitialConditions.Membrane([[domain[0] * 0.5,
                                              domain[1] * 0.5,
                                              domain[2] * 0.5,
                                              1.0, 0.0, 0.0, 0.0]])
    u.registerParticleVector(pv_rbc, ic_rbc)

    force_scale_ = 0.25 * ureg.pN # arbitrary
    length_scale_ = np.sqrt(params.A0 / A0)
    time_scale_ = 1 * ureg.s # arbitrary
    mass_scale_ = (force_scale_ / length_scale_ * time_scale_**2).to(ureg.kg)

    force = float(force_ / force_scale_)

    rbc_params = params.get_params(length_scale=length_scale_,
                                   time_scale=time_scale_,
                                   mass_scale=mass_scale_,
                                   mesh=mesh_ini)

    rbc_params.kBT = 5e-4 * rbc_params.bending_modulus()
    rbc_params.gamma = 200.0 # for killing oscillations

    tend   = 150.0
    dt0    = rbc_params.get_max_dt(mass=mass)
    substeps = 500
    dt     = substeps * dt0

    int_rbc = mir.Interactions.MembraneForces("int_rbc", **rbc_params.to_interactions(), stress_free=True)
    u.registerInteraction(int_rbc)
    vv = mir.Integrators.SubStep('vv', substeps, [int_rbc])
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_rbc)
    # u.setInteraction(int_rbc, pv_rbc, pv_rbc)

    dc_ = 2 * ureg.um
    dc = float(dc_ / length_scale_)

    forces = compute_micro_beads_forces(mesh_ini, contact_diameter=dc, bead_force=force).tolist()

    u.registerPlugins(mir.Plugins.createMembraneExtraForce("stretchForce", pv_rbc, forces))

    if dump:
        ply_path = f"ply_f_{float(force_/ureg.pN)}"
        u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, int(tend/50/dt), ply_path))

    u.run(int(tend/dt), dt)

    is_master = u.isMasterTask()
    Dshort, Dlong = 0, 0

    final_mesh = copy.deepcopy(mesh_ini)

    if is_master:
        rbc_pos = pv_rbc.getCoordinates()
        final_mesh.vertices = rbc_pos
        final_mesh = center_align_mesh(final_mesh)
        Dlong, Dshort = compute_diameters(final_mesh)

        Dlong  *= length_scale_
        Dshort *= length_scale_
    del u

    return is_master, final_mesh, Dshort, Dlong


def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_ini', type=str, help='The initial mesh.')
    parser.add_argument('mesh_ref', type=str, help='The mesh of stress free state.')
    parser.add_argument('exp_file', type=str, help='The csv file containing the experimental values.')
    parser.add_argument('--out', type=str, default="results.csv")
    parser.add_argument('--dump', action='store_true', default=False, help="Will dump ply files if set to True.")
    args = parser.parse_args(argv)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_address = MPI._addressof(comm)

    ureg = pint.UnitRegistry()


    # see https://github.com/mikedh/trimesh/issues/338
    mesh_ini = trimesh.load_mesh(args.mesh_ini, process=False)
    mesh_ref = trimesh.load_mesh(args.mesh_ref, process=False)

    df = pd.read_csv(args.exp_file)

    D0s = list()
    D1s = list()

    params = JuelicherLimRBCDefaultParams(ureg)

    for force in df['force']:
        force_ = force * ureg.pN
        is_master, _, D0, D1 = run_stretching(ureg=ureg,
                                              comm_address=comm_address,
                                              mesh_ini=mesh_ini,
                                              mesh_ref=mesh_ref,
                                              force_=force_,
                                              params=params,
                                              dump=args.dump)
        if is_master:
            D0 = D0.to(ureg.um).magnitude
            D1 = D1.to(ureg.um).magnitude

            print(force, D0, D1)
            sys.stdout.flush()

        D0s.append(D0)
        D1s.append(D1)

    df['D0sim'] = D0s
    df['D1sim'] = D1s

    if is_master:
        df.to_csv(args.out, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
