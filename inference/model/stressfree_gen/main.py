#!/usr/bin/env python

import copy
import numpy as np
import pint
import trimesh
import sys

from dpdprops import (MembraneParams,
                      LimParams,
                      JuelicherParams)
import mirheo as mir


def rescale_mesh_by_area(mesh: trimesh.Trimesh,
                         A: float):
    """
    Args:
        mesh: The mesh to rescale
        A: the target area
    Return:
        the mesh rescaled such that its area is the input one
    """
    scale = np.sqrt(A / mesh.area)
    mesh.vertices *= scale
    return mesh

def generate_stressfree_mesh(*,
                             ureg,
                             mesh_ini: trimesh.Trimesh,
                             mesh_ref: trimesh.Trimesh,
                             reduced_volume: float,
                             kb: float=1,
                             RA: float=1,
                             comm_address: int=0,
                             dump: bool=False):
    """
    Args:
        ureg: Unit registry.
        mesh_ini: The initial mesh (will be rescaled).
        mesh_ref: The reference mesh (will be rescaled). Assumed to be a sphere for this setup.
        reduced_volume: The reduced volume of the cell, between 0 qnd 1
        kb: The bending modulus in simulation units (sets the energy scale)
        RA: The equivalent radius of the cell in simulation units (sets the length scale)
        comm_address: if set, the address of the mpi communicator to use. Otherwise use the comm_world communicator.
        dump: if True, dump stats and ply files.
    """

    if reduced_volume < 0 or reduced_volume > 1:
        raise ValueError(f"unexpected reduced volume value {reduced_volume}")

    assert len(mesh_ref.vertices) == len(mesh_ini.vertices)

    # Physical parameters

    kb_ = 2e-19 * ureg.J
    mu_ = 0.2 * kb_ / (1 * ureg.um**2)
    ka_ = kb_ / (1 * ureg.um**2)

    ## adjusted; value from Athena
    # TODO: ask where this is from
    kA_ = 4.59e-3 * ureg.J / (ureg.m**2)
    kV_ = 1.376e3 * ureg.J / (ureg.m**3)

    A0_ = 140 * ureg.um**2
    RA_ = np.sqrt(A0_ / (4 * np.pi))
    V0_ = reduced_volume * 4 * np.pi / 3 * RA_**3

    length_scale_ = (RA_ / RA).to(ureg.um)
    energy_scale_ = (kb_ / kb).to(ureg.J)
    force_scale_ = (energy_scale_ / length_scale_).to(ureg.pN)

    assert length_scale_.check('[length]')
    assert force_scale_.check('[force]')
    assert energy_scale_.check('[energy]')

    # Simulation parameters

    A0 = float(A0_ / length_scale_**2)
    V0 = float(V0_ / length_scale_**3)
    mu = float(mu_ / force_scale_ * length_scale_)
    ka = float(ka_ / force_scale_ * length_scale_)

    kA = float(kA_ / energy_scale_ * length_scale_**2)
    kV = float(kV_ / energy_scale_ * length_scale_**3)

    kBT = 1e-4 * kb
    gamma = 200 # arbitrary, defines the time scale (not important here)
    mass=1 # arbitrary, does not matter because there are no dynamics

    mesh_ini = rescale_mesh_by_area(mesh_ini, A0)
    mesh_ref = rescale_mesh_by_area(mesh_ref, A0)

    rbc_params = MembraneParams(area=A0,
                                volume=V0,
                                ka=kA,
                                kv=kV,
                                gamma=gamma,
                                kBT=kBT,
                                nv=len(mesh_ini.vertices),
                                bending_params=JuelicherParams(kb=kb, C0=0, kad=0, DA0=0),
                                shear_params=LimParams(ka=ka, mu=mu))


    domain = 5 * np.ptp(mesh_ini.vertices, axis=0)
    ranks = (1,1,1)

    u = mir.Mirheo(ranks, domain, debug_level=0, log_filename='log', no_splash=True, comm_ptr=comm_address)

    mesh_rbc = mir.ParticleVectors.MembraneMesh(vertices=mesh_ini.vertices,
                                                stress_free_vertices=mesh_ref.vertices,
                                                faces=mesh_ini.faces)
    pv_rbc = mir.ParticleVectors.MembraneVector("rbc", mass=mass, mesh=mesh_rbc)
    ic_rbc = mir.InitialConditions.Membrane([(domain/2).tolist() + [1.0, 0.0, 0.0, 0.0]])
    u.registerParticleVector(pv_rbc, ic_rbc)

    int_rbc = mir.Interactions.MembraneForces("int_rbc", **rbc_params.to_interactions(), stress_free=True)
    u.registerInteraction(int_rbc)

    substeps = 1000

    vv_rbc = mir.Integrators.SubStep("vv", substeps=substeps, fastForces=[int_rbc])
    u.registerIntegrator(vv_rbc)
    u.setIntegrator(vv_rbc, pv_rbc)

    dt_real = rbc_params.get_max_dt(mass=mass) / 5

    tend = 4 * gamma / mu
    dt = dt_real * substeps
    nsteps = int(tend/dt)


    t_dump_every = tend / 5
    dump_every = int(t_dump_every/dt)

    if dump:
        u.registerPlugins(mir.Plugins.createStats('stats', every=dump_every, filename='stats.csv'))
        u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, "ply_S0/"))

    u.run(nsteps, dt)

    is_master = u.isMasterTask()
    mesh = copy.deepcopy(mesh_ini)

    if is_master:
        mesh.vertices = pv_rbc.getCoordinates()

    del u

    return is_master, mesh



def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Generate the stress-free shape as described in Lim et al. 2008, p.117')
    parser.add_argument('--mesh_ini', type=str, help="Initial mesh.")
    parser.add_argument('--mesh_ref', type=str, help="Reference mesh (spherical).")
    parser.add_argument('--reduced-volume', type=float, default=0.96, help="Desired reduced volume.")
    parser.add_argument('--out', type=str, default=None, help="The name of the resulting mesh file.")
    args = parser.parse_args(argv)

    mesh_ini = trimesh.load(args.mesh_ini, process=False)
    mesh_ref = trimesh.load(args.mesh_ref, process=False)

    ureg = pint.UnitRegistry()

    is_master, mesh = generate_stressfree_mesh(ureg=ureg,
                                               reduced_volume=args.reduced_volume,
                                               mesh_ini=mesh_ini,
                                               mesh_ref=mesh_ref)
    if is_master:
        if args.out is not None:
            mesh.export(args.out)



if __name__ == '__main__':
    main(sys.argv[1:])
