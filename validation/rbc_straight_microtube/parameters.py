#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pint
import trimesh
import sys

import dpdprops

@dataclass
class PipeFlowParams:
    rc: float
    m: float
    nd: float
    RA: float
    kBT: float
    L: float
    R: float
    Vmax: float
    pressure_gradient: float
    mesh_ini: trimesh.Trimesh
    mesh_ref: trimesh.Trimesh
    rbc_params: dpdprops.MembraneParams
    dpd_oo: dpdprops.DPDParams
    dpd_ii: dpdprops.DPDParams
    dpd_io: dpdprops.DPDParams
    dpd_rbco: dpdprops.DPDParams
    dpd_rbci: dpdprops.DPDParams
    max_contact_force: float
    length_scale_: pint.Quantity
    time_scale_: pint.Quantity
    mass_scale_: pint.Quantity
    shear_scale_factor: float

def rescale_mesh(mesh, RA: float):
    RA_current = (mesh.area / (4 * np.pi)) ** (1/2)
    scale = RA / RA_current
    mesh.vertices *= scale


def create_parameters(ureg,
                      pressure_gradient_,
                      Vmax_,
                      L_,
                      R_,
                      mesh_ini,
                      mesh_ref,
                      RA: float,
                      Re: float,
                      visc_ratio: float,
                      mu: float=20,
                      verbose: bool=True):

    @ureg.wraps(None, ureg.dimensionless)
    def to_sim(a):
        return a

    assert pressure_gradient_.check('[pressure] / [length]')
    assert Vmax_.check('[velocity]')
    assert L_.check('[length]')
    assert R_.check('[length]')

    rescale_mesh(mesh_ini, RA)
    rescale_mesh(mesh_ref, RA)

    # rbc = dpdprops.JuelicherLimRBCDefaultParams(ureg,
    #                                             A0=140 * ureg.um**2,
    #                                             V0=100 * ureg.um**3,
    #                                             ka=5 * ureg.uN / ureg.m,
    #                                             mu=2.5 * ureg.uN / ureg.m,
    #                                             kappab=2e-19 * ureg.J,
    #                                             b2=0.75,
    #                                             m0_bar=10,
    #                                             CADE=2/np.pi)
    rbc = dpdprops.JuelicherLimRBCDefaultParams(ureg,
                                                A0=135 * ureg.um**2,
                                                V0=94 * ureg.um**3,
                                                ka=3.94 * ureg.uN / ureg.m,
                                                mu=3.94 * ureg.uN / ureg.m,
                                                kappab=2.83e-19 * ureg.J,
                                                b2=1.81,
                                                m0_bar=0,
                                                CADE=0,
                                                eta_m=0.42 * ureg.Pa * ureg.s * ureg.um)
    # rbc = dpdprops.JuelicherLimRBCDefaultParams(ureg)

    # physical constants
    RA_ = np.sqrt(rbc.A0 / (4 * np.pi))
    etao_ = 1.003e-3 * ureg.Pa * ureg.s
    rho_ = 1e3 * ureg.kg / ureg.m**3
    nuo_ = etao_ / rho_
    kB_ = 1.38064852e-23 * ureg.m**2 * ureg.kg / (ureg.s**2 * ureg.K)
    T_ = ureg.Quantity(37, ureg.degC).to('kelvin')
    kBT_ = kB_ * T_

    # physical parameters
    mean_shear_ = Vmax_ / R_

    # dimless numbers in "real" world
    Re_ = to_sim(mean_shear_ * RA_**2 / nuo_)
    Ca_ = to_sim(mean_shear_ * RA_ * etao_ / rbc.mu)
    E_ = to_sim(rbc.eta_m / (RA_ * etao_))
    kb_kBT_ = to_sim(rbc.kappab / kBT_)

    # go to simulation world
    rc = 1
    nd = 10
    m = 1
    rho = m * nd
    a_rc_kBTinv = 100

    Ca = Ca_
    E = E_
    kb_kBT = kb_kBT_

    # solve for eta, mean_shear and etam to satisfy Re, Ca and E
    etao = np.sqrt(Ca * mu * rho * RA / Re)
    mean_shear = Ca * mu / (RA * etao)
    eta_m = E * RA * etao


    f = np.sqrt(Re / Re_)

    length_scale_ = RA_ / RA
    time_scale_ = mean_shear / (f * mean_shear_)
    mass_scale_ = (rho_ * length_scale_**3 / rho).to(ureg.kg)

    force_scale_ = length_scale_ * mass_scale_ / time_scale_**2

    R = to_sim(R_ / length_scale_)
    L = to_sim(L_ / length_scale_)
    Vmax = R * mean_shear


    assert length_scale_.check('[length]')
    assert force_scale_.check('[force]')
    assert mass_scale_.check('[mass]')
    assert time_scale_.check('[time]')

    assert abs(Re - RA**2 * mean_shear * rho / etao) < 1e-2

    # RBC parameters

    rbc_prms = rbc.get_params(mesh=mesh_ini,
                              length_scale=length_scale_,
                              time_scale=time_scale_,
                              mass_scale=mass_scale_)

    rbc_prms.set_viscosity(eta_m)


    # fluid parameters

    kBT = rbc_prms.bending_modulus() / kb_kBT

    Cs = dpdprops.sound_speed(a=a_rc_kBTinv*kBT/rc,
                              nd=nd,
                              mass=m,
                              rc=rc,
                              kBT=kBT)

    U = RA * mean_shear
    Ma = U / Cs

    dpd_oo_prms = dpdprops.create_dpd_params_from_Re_Ma(Re=Re, Ma=Ma, U=U, L=RA,
                                                        nd=nd, rc=rc, mass=m,
                                                        a_rc_kBTinv=a_rc_kBTinv)

    nuo = dpd_oo_prms.kinematic_viscosity()
    nui = nuo * visc_ratio

    dpd_ii_prms = dpdprops.create_dpd_params_from_props(kinematic_viscosity=nui,
                                                        sound_speed=dpd_oo_prms.sound_speed(),
                                                        nd=nd, rc=rc, mass=m, a_rc_kBTinv=a_rc_kBTinv)

    dpd_io_prms = deepcopy(dpd_ii_prms)
    dpd_io_prms.gamma = 0 # keep only the repulsion force.


    # FSI params

    nd_2D_membrane = len(mesh_ini.vertices) / mesh_ini.area
    dpd_rbco_prms = dpdprops.create_fsi_dpd_params(fluid_params=dpd_oo_prms, nd_membrane=nd_2D_membrane)
    dpd_rbci_prms = dpdprops.create_fsi_dpd_params(fluid_params=dpd_ii_prms, nd_membrane=nd_2D_membrane)


    # body force
    pressure_gradient = to_sim(pressure_gradient_ / (force_scale_ / length_scale_**3))

    # Wall repulsion forces
    max_contact_force = dpd_oo_prms.a * 200

    if verbose:
        print(f"length_scale = {length_scale_.to(ureg.um)}")
        print(f"force_scale = {force_scale_.to(ureg.pN)}")
        print(f"time_scale = {time_scale_.to(ureg.s)}")
        print(f"mass_scale = {mass_scale_.to(ureg.kg)}")

        print(f"mu = {mu}")
        print(f"Vmax = {Vmax}")
        print(f"etao = {etao}")
        print(f"eta_m = {eta_m}")

        print(f"Re = {Re_} (phys), {Re} (sim)")
        print(f"Ca = {Ca_} (phys), {Ca} (sim)")
        print(f"FvK = {float(rbc.mu * RA_**2 / rbc.kappab)}")
        print(f"E = {E_} (phys), {E} (sim)")
        print(f"Ma = {Ma} (sim)")
        print(rbc_prms)

        print(f"dpd_oo = {dpd_oo_prms}")
        print(f"dpd_ii = {dpd_ii_prms}")
        print(f"dpd_io = {dpd_io_prms}")

        print(f"R = {R_} ({R})")
        print(f"L = {L_} ({L})")

        print(f"pressure_gradient = {pressure_gradient}")

        sys.stdout.flush()

    p = PipeFlowParams(rc=rc, m=m, nd=nd, RA=RA,
                       kBT=kBT, L=L, R=R, Vmax=Vmax,
                       pressure_gradient=pressure_gradient,
                       mesh_ini=mesh_ini,
                       mesh_ref=mesh_ref,
                       rbc_params=rbc_prms,
                       dpd_oo=dpd_oo_prms,
                       dpd_ii=dpd_ii_prms,
                       dpd_io=dpd_io_prms,
                       dpd_rbco=dpd_rbco_prms,
                       dpd_rbci=dpd_rbci_prms,
                       max_contact_force=max_contact_force,
                       length_scale_=length_scale_,
                       time_scale_=time_scale_,
                       mass_scale_=mass_scale_,
                       shear_scale_factor=f)
    return p


def dump_parameters(p: 'PipeFlowParams',
                    filename: str):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)

def load_parameters(filename: str):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Run RBCs flowing in a capillary pipe.')
    parser.add_argument('--rbc-res', type=int, default=4, help="Initial RBC mesh.")
    parser.add_argument('--L', type=str, default="50um", help="Length of the pipe.")
    parser.add_argument('--R', type=str, default="10um", help="Radius of the pipe.")
    parser.add_argument('--Re', type=float, default=0.5, help="Simulation Reynolds number.")
    parser.add_argument('--RA', type=float, default=4, help="Reduced radius of the RBC, in simulation units. This sets the length scale.")
    parser.add_argument('--Vmax', type=str, default="1_cm_per_s", help="Maximum velocity.")
    parser.add_argument('--pressure-gradient', type=str, required=True, help="Pressure gradient (must be a pint expression)")
    parser.add_argument('--C', type=float, default=5, help="Viscosity ratio between inner and outer solvent")
    parser.add_argument('--eta-out', type=str, default='1e-3_Pa_s', help="Viscosity of the outer fluid (must be a pint expression)")
    parser.add_argument('--out-params', type=str, default="parameters.pkl", help="Save all simulation parameters to this file.")
    args = parser.parse_args(argv)

    # v = 0.94
    # mesh_ini = dpdprops.load_equilibrium_mesh(subdivisions=args.rbc_res, reduced_volume=v)
    # mesh_ref = dpdprops.load_stress_free_mesh(subdivisions=args.rbc_res, reduced_volume=v)
    mesh_ini = dpdprops.load_equilibrium_mesh(subdivisions=args.rbc_res)
    mesh_ref = dpdprops.load_stress_free_mesh(subdivisions=args.rbc_res)

    ureg = pint.UnitRegistry()

    def to_ureg(s: str):
        # we avoid "/" and spaces so that directory names are not weird
        if '/' in s or ' ' in s:
            raise ValueError(f"'{s}' should not contain '/' or spaces, otherwise it will create directories with undesired names. Use 'per' and '_' instead.")
        return ureg(s.replace('_', ' '))

    pg_ = to_ureg(args.pressure_gradient)
    Vmax_ = to_ureg(args.Vmax)
    L_ = to_ureg(args.L)
    R_ = to_ureg(args.R)
    etao_ = to_ureg(args.eta_out)

    p = create_parameters(ureg,
                          pressure_gradient_=pg_,
                          Vmax_=Vmax_,
                          L_=L_,
                          R_=R_,
                          Re=args.Re,
                          RA=args.RA,
                          visc_ratio=args.C,
                          mesh_ini=mesh_ini,
                          mesh_ref=mesh_ref)
    dump_parameters(p, args.out_params)



if __name__ == '__main__':
    main(sys.argv[1:])
