#!/usr/bin/env python

import copy
import numpy as np
import os
import pint
from scipy.optimize import least_squares
import sys
import trimesh

import mirheo as mir
from dpdprops import JuelicherLimRBCDefaultParams

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, ".."))

from tools import (center_align_mesh,
                   compute_diameters)


def rescale_by_area(mesh, A):
    mesh.vertices *= np.sqrt(A / mesh.area)
    return mesh

def fit_tc(t, LW):
    """
    Fit the function (eq 10) from Hochmuth 1979 to the data via least squares and compute tc.
    Arguments:
        t: Array of times.
        LW array of L/W.
    Returns:
        tc The characteristic time of the relaxation.
    """
    LW0 = LW[0]

    def LW_theory(t, tc, LWinf):
        L = (LW0 + LWinf) / (LW0 - LWinf)
        return LWinf * (L + np.exp(-t/tc)) / (L - np.exp(-t/tc))

    def f(x):
        tc, LWinf = x
        LWth = LW_theory(t, tc, LWinf)
        return LW - LWth

    x0 = [0.15, 1]
    res = least_squares(f, x0)

    tc, LWinf = res.x

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(t, LW, '+k', label='Simulation')
        ax.plot(t, LW_theory(t, tc, LWinf), '-k', label='fit')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$L/W$')
        ax.legend()
        plt.show()

    return tc


def run_relaxation(*,
                   ureg,
                   comm_address,
                   mesh_ini,
                   mesh_ref,
                   params,
                   RA: float=1,
                   dump: bool=False):
    """
    Parameters:
        ureg: the unit registry
        comm_address: the address of the MPI communicator
        mesh_ini: Initial mesh
        mesh_ref: stress free mesh
        params: The RBC parameters (see dpdprops).
        RA: equivalent radius of the membrane in simulation units (sets the length scale)
        dump: if True, dump ply files.
    """

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

    length_scale_ = np.sqrt(params.A0 / A0)
    time_scale_ = 0.1 * ureg.s # arbitrary
    mass_scale_ = 1e-11 * ureg.kg # arbitrary

    rbc_params = params.get_params(length_scale=length_scale_,
                                   time_scale=time_scale_,
                                   mass_scale=mass_scale_,
                                   mesh=mesh_ini)

    rbc_params.kBT = 5e-4 * rbc_params.bending_modulus()

    tc = rbc_params.get_viscosity() / rbc_params.shear_modulus()

    # dt_el = rbc_params.get_max_dt_elastic(mass=mass)
    # dt_visc = rbc_params.get_max_dt_visc(mass=mass)
    dt0    = rbc_params.get_max_dt(mass=mass) / 2

    ndumps = 100
    tend_ = 1 * ureg.s
    tend = float(tend_ / time_scale_)
    t_dump_every = tend / ndumps

    substeps = int(t_dump_every / dt0) // 10
    nsweeps = 0
    dt     = substeps * dt0

    if nsweeps > 0:
        int_rbc = mir.Interactions.MembraneForces("int_rbc", **rbc_params.to_interactions_zero_visc(), stress_free=True)
        u.registerInteraction(int_rbc)
        vv = mir.Integrators.SubStepShardlowSweep('vv', substeps, int_rbc, **rbc_params.to_viscous(), nsweeps=nsweeps)
    else:
        int_rbc = mir.Interactions.MembraneForces("int_rbc", **rbc_params.to_interactions(), stress_free=True)
        u.registerInteraction(int_rbc)
        vv = mir.Integrators.SubStep('vv', substeps, [int_rbc])

    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_rbc)
    # u.setInteraction(int_rbc, pv_rbc, pv_rbc)

    dump_every = int(t_dump_every / dt)
    t_dump_every = dump_every * dt

    if dump:
        u.registerPlugins(mir.Plugins.createDumpMesh("rbc_dump", pv_rbc, dump_every, 'ply_relax'))

    L = []
    W = []

    is_master = u.isMasterTask()

    for i in range(ndumps):
        u.run(dump_every, dt)

        curr_mesh = copy.deepcopy(mesh_ini)
        if is_master:
            rbc_pos = pv_rbc.getCoordinates()
            curr_mesh.vertices = rbc_pos
            curr_mesh = center_align_mesh(curr_mesh)
            Dlong, Dshort = compute_diameters(curr_mesh)

            L += [Dlong]
            W += [Dshort]

    del u

    # compute the characteristic time tc_
    tc_ = 0

    if is_master:
        L = np.array(L)
        W = np.array(W)
        LW = L / W
        t = np.array([i * t_dump_every for i in range(ndumps)])
        tc_ = fit_tc(t, LW) * time_scale_

    return is_master, tc_



def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_ini', type=str, help='The initial mesh.')
    parser.add_argument('mesh_ref', type=str, help='The mesh of stress free state.')
    args = parser.parse_args(argv)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm_address = MPI._addressof(comm)

    ureg = pint.UnitRegistry()

    # see https://github.com/mikedh/trimesh/issues/338
    mesh_ini = trimesh.load_mesh(args.mesh_ini, process=False)
    mesh_ref = trimesh.load_mesh(args.mesh_ref, process=False)

    params = JuelicherLimRBCDefaultParams(ureg)

    is_master, tc_ = run_relaxation(ureg=ureg,
                                    comm_address=comm_address,
                                    mesh_ini=mesh_ini,
                                    mesh_ref=mesh_ref,
                                    params=params,
                                    dump=False)

    print(f"found tc = {tc_} (eta_m/mu = {(params.eta_m / params.mu).to(ureg.s)})")


if __name__ == '__main__':
    main(sys.argv[1:])
