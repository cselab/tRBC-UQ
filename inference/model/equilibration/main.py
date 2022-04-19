#!/usr/bin/env python

import copy
import numpy as np
import os
import pint
import sys
import trimesh

import mirheo as mir
from dpdprops import JuelicherLimRBCDefaultParams

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from tools import center_align_mesh

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


def compute_D_hmin_hmax_(mesh0: trimesh.Trimesh):
    """
    Compute the diameter D, maximum thickness hmax and thickness at the center hmin of a cell.

    Args:
        mesh: The mesh of the cell.
    Return:
        D, hmin, hmax
    """
    mesh = copy.deepcopy(mesh0)
    mesh = center_align_mesh(mesh)

    x = mesh.vertices[:,0]
    y = mesh.vertices[:,1]
    z = mesh.vertices[:,2]

    D = np.mean(np.ptp(mesh.vertices[:,:2], axis=0))

    # assume the cell is radially symmetric; we measure the mean height along r in bins
    # and then take the minimum and maximum
    r = np.sqrt(x**2 + y**2)

    nv = len(x)
    bin_edges = np.linspace(0, D/2, nv//100, endpoint=True)
    bin_r = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_h = []

    for r0, r1 in zip(bin_edges[:-1], bin_edges[1:]):
        idx = np.argwhere(np.logical_and(r >= r0, r < r1))
        if len(idx):
            h = np.ptp(z[idx])
        else:
            h = 0
        bin_h.append(h)

    bin_h = np.array(bin_h)

    hmax = np.max(bin_h)
    # hmin is the minimum height of the cell in the inside region,
    # i.e. we must filter out the particles that have a larger radial coordinate
    # than those at hmax.
    idmax = np.argmax(np.abs(bin_h))
    rhmax = bin_r[idmax]
    idx = np.argwhere(bin_r <= rhmax)
    hmin = np.min(bin_h[idx])

    return D, hmin, hmax


def compute_D_hmin_hmax(mesh0: trimesh.Trimesh):
    """
    Compute the diameter D, maximum thickness hmax and thickness at the center hmin of a cell.

    Args:
        mesh: The mesh of the cell.
    Return:
        D, hmin, hmax
    """
    mesh = copy.deepcopy(mesh0)
    mesh = center_align_mesh(mesh)

    D = np.mean(np.ptp(mesh.vertices[:,:2], axis=0))

    n = 2000
    ray_origins = np.array([[x, 0, -50] for x in np.linspace(0, D/2, n)])
    ray_directions = np.array([[0, 0, 1]] * n)

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True)

    r = []
    h = []
    for i in np.unique(index_ray):
        idx = np.argwhere(index_ray == i)
        h.append(np.ptp(locations[idx,2]))
        r.append(locations[idx,0][0][0])

    r = np.array(r)
    h = np.array(h)
    idx = np.argsort(r)
    r = r[idx]
    h = h[idx]

    hmax = np.max(h)
    # hmin is the minimum height of the cell in the inside region,
    # i.e. we must filter out the particles that have a larger radial coordinate
    # than those at hmax.
    idmax = np.argmax(h)
    rhmax = r[idmax]
    idx = np.argwhere(r <= rhmax)
    hmin = np.min(h[idx])

    return D, hmin, hmax






def run_equilibration(*,
                      ureg,
                      mesh_ini: trimesh.Trimesh,
                      mesh_ref: trimesh.Trimesh,
                      params,
                      RA: float=1,
                      comm_address: int=0,
                      dump: bool=False):
    """
    Args:
        ureg: unit registry
        mesh_ini: The initial mesh (will be rescaled)
        mesh_ref: The reference mesh (will be rescaled)
        params: The rbc params (see dpdprops).
        RA: The equivalent radius of the cell in simulation units (sets the length scale)
        comm_address: if set, the address of the mpi communicator to use. Otherwise use the comm_world communicator.
        dump: if True, dumps ply files and stats.
    """

    assert len(mesh_ref.vertices) == len(mesh_ini.vertices)

    mass = 1.0
    gamma = 500

    A0 = 4 * np.pi * RA**2

    mesh_ini = rescale_mesh_by_area(mesh_ini, A0)
    mesh_ref = rescale_mesh_by_area(mesh_ref, A0)

    length_scale_ = (np.sqrt(params.A0 / A0)).to(ureg.um)
    energy_scale_ = 1e-19 * ureg.J # arbitrary
    force_scale_ = (energy_scale_ / length_scale_).to(ureg.pN)
    time_scale_ = 1 * ureg.s # arbitrary
    mass_scale_ = force_scale_ * time_scale_**2 / length_scale_

    assert length_scale_.check('[length]')
    assert force_scale_.check('[force]')
    assert energy_scale_.check('[energy]')
    assert time_scale_.check('[time]')
    assert mass_scale_.check('[mass]')

    # Simulation parameters


    rbc_params = params.get_params(length_scale=length_scale_,
                                   time_scale=time_scale_,
                                   mass_scale=mass_scale_,
                                   mesh=mesh_ini)

    rbc_params.gamma = gamma
    rbc_params.kBT = 1e-4 * rbc_params.bending_modulus()

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

    dt_real = rbc_params.get_max_dt(mass=mass)

    tend = 100 * gamma / rbc_params.shear_modulus()
    dt = dt_real * substeps
    nsteps = int(tend/dt)

    t_dump_every = tend / 5
    dump_every = int(t_dump_every/dt)

    if dump:
        u.registerPlugins(mir.Plugins.createStats('stats', every=dump_every, filename='stats.csv'))
        u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", pv_rbc, dump_every, "ply_eq/"))

    u.run(nsteps, dt)

    is_master = u.isMasterTask()
    mesh = copy.deepcopy(mesh_ini)
    D, hmax, hmin = None, None, None

    if is_master:
        mesh.vertices = pv_rbc.getCoordinates()
        D, hmin, hmax = compute_D_hmin_hmax(mesh)
        D *= length_scale_
        hmin *= length_scale_
        hmax *= length_scale_

    del u

    return is_master, mesh, D, hmin, hmax



def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Generate the stress-free shape as described in Lim et al. 2008, p.117')
    parser.add_argument('--mesh-ini', type=str, required=True, help="Initial mesh.")
    parser.add_argument('--mesh-ref', type=str, required=True, help="Reference mesh (spherical).")
    parser.add_argument('--out', type=str, default=None, help="The name of the resulting mesh file.")
    args = parser.parse_args(argv)

    mesh_ini = trimesh.load(args.mesh_ini, process=False)
    mesh_ref = trimesh.load(args.mesh_ref, process=False)

    ureg = pint.UnitRegistry()

    params = JuelicherLimRBCDefaultParams(ureg)

    is_master, mesh, D, hmin, hmax = run_equilibration(ureg=ureg,
                                                       mesh_ini=mesh_ini,
                                                       mesh_ref=mesh_ref,
                                                       params=params,
                                                       dump=True)
    if is_master:
        print(D, hmin, hmax)
        if args.out is not None:
            mesh.export(args.out)



if __name__ == '__main__':
    main(sys.argv[1:])
