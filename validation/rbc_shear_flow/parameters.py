#!/usr/bin/env python

from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pint
import sys
import trimesh

import dpdprops

@dataclass
class RBCShearParams:
    rc: float
    m: float
    nd: float
    RA: float
    kBT: float
    L: float
    shear_rate: float
    mesh_ini: trimesh.Trimesh
    mesh_ref: trimesh.Trimesh
    rbc_params: dpdprops.MembraneParams
    dpd_oo: dpdprops.DPDParams
    dpd_ii: dpdprops.DPDParams
    dpd_io: dpdprops.DPDParams
    dpd_rbco: dpdprops.DPDParams
    dpd_rbci: dpdprops.DPDParams
    length_scale_: pint.Quantity
    time_scale_: pint.Quantity
    mass_scale_: pint.Quantity
    t_dump_every: float
    scaling_factor: float


def rotate_mesh(mesh):
    """
    Rotate a mesh by 90 degreesin the yz plane
    """
    v = np.array(mesh.vertices, copy=True)
    mesh.vertices[:,1] = -v[:,2]
    mesh.vertices[:,2] =  v[:,1]

def rescale_mesh(mesh, RA: float):
    RA_current = (mesh.area / (4 * np.pi)) ** (1/2)
    scale = RA / RA_current
    mesh.vertices *= scale

def create_parameters(*,
                      ureg,
                      mesh_ini,
                      mesh_ref,
                      rbc_params: dpdprops.JuelicherLimRBCDefaultParams,
                      Re: float,
                      RA: float,
                      etao_: pint.Quantity,
                      etai_: pint.Quantity,
                      shear_rate_: pint.Quantity,
                      verbose: bool=False):
    """
    Argument:
        ureg: Unit registry
        mesh_ini: Initial rbc mesh
        mesh_ref: Stress free rbc mesh
        rbc_params: The parameters of the RBC, in physical units
        Re: The simulation Reynolds number (can be higher than the real one).
        RA: the equivalent radius of the RBC in simulation units (sets the resolution).
        etao_: The viscosity of the solvent at room temperature.
        etai_: The viscosity of the cytosol at room temperatur.e
        shear_rate_: The shear rate.
        verbose: if True, will print parameters to stdout

    Return:
        an instance of RBCShearParams
    """

    # Physical constants
    kB_ = 1.380649e-23 * ureg.J / ureg.K
    T_ = ureg.Quantity(20, ureg.degC).to('kelvin') # room temperature
    kBT_ = (kB_ * T_).to(ureg.J)

    rho_ = 1e3 * ureg.kg / ureg.m**3

    # Simulation parameters
    mass=1
    rc=1
    num_density=10 # TODO: this can be tuned to match an input Mach number.
    a_rc_kBTinv=100
    L = 10 * RA # based on convergence studies
    shear_rate = 0.1 # set the time scale


    # compute the dimensionless numbers (physical space)
    RA_ = dpdprops.equivalent_sphere_radius(area=rbc_params.A0)
    U_ = shear_rate_ * RA_

    Re_ = float(U_ * RA_ * rho_ / etao_)
    CA_ = float(U_ * etao_ / rbc_params.mu)
    kb_kBT_ = float(rbc_params.kappab / kBT_)
    visc_ratio_ = float(etai_ / etao_)

    Ca = CA_
    kb_kBT = kb_kBT_
    visc_ratio = visc_ratio_

    # scaling trick: multiplies shear rate and divides viscosity in the simulation units
    scaling_factor = np.sqrt(Re / Re_)

    rbc_params.kBT = kBT_

    rotate_mesh(mesh_ini)
    rescale_mesh(mesh_ini, RA)
    rescale_mesh(mesh_ref, RA)

    U = shear_rate * RA
    nu_oo = U * RA / Re
    mu = U * num_density * mass * nu_oo / Ca

    length_scale_ = (RA_ / RA).to(ureg.um)
    force_scale_ = (rbc_params.mu / mu * length_scale_).to(ureg.N)
    time_scale_ = (shear_rate / (shear_rate_ * scaling_factor)).to(ureg.s)
    mass_scale_ = (force_scale_ * time_scale_**2 / length_scale_).to(ureg.kg)

    # Go to simulation units

    rbc_prms = rbc_params.get_params(length_scale=length_scale_,
                                     time_scale=time_scale_,
                                     mass_scale=mass_scale_,
                                     mesh=mesh_ini)

    kBT = rbc_prms.bending_modulus() / kb_kBT
    rbc_prms.set_viscosity(rbc_prms.get_viscosity() / scaling_factor)

    Cs = dpdprops.sound_speed(a=a_rc_kBTinv*kBT/rc,
                              nd=num_density,
                              mass=mass,
                              rc=rc,
                              kBT=kBT)
    Ma = U / Cs

    dpd_oo_prms = dpdprops.create_dpd_params_from_Re_Ma(Re=Re, Ma=Ma, U=U, L=RA,
                                                        nd=num_density, rc=rc,
                                                        mass=mass, a_rc_kBTinv=a_rc_kBTinv)

    nuo = dpd_oo_prms.kinematic_viscosity()
    nui = nuo * visc_ratio

    dpd_ii_prms = dpdprops.create_dpd_params_from_props(kinematic_viscosity=nui,
                                                        sound_speed=dpd_oo_prms.sound_speed(),
                                                        nd=num_density, rc=rc,
                                                        mass=mass, a_rc_kBTinv=a_rc_kBTinv)

    dpd_io_prms = deepcopy(dpd_oo_prms)
    dpd_io_prms.gamma = 0

    nd_membrane = len(mesh_ini.vertices) / mesh_ini.area

    dpd_rbco_prms = dpdprops.create_fsi_dpd_params(fluid_params=dpd_oo_prms,
                                                   nd_membrane=nd_membrane)
    dpd_rbci_prms = dpdprops.create_fsi_dpd_params(fluid_params=dpd_ii_prms,
                                                   nd_membrane=nd_membrane)

    domain = (L, L + 4*rc, L)

    np.testing.assert_array_equal(mesh_ini.faces,
                                  mesh_ref.faces,
                                  err_msg="The input meshes must have the same topology.")

    if verbose:
        print(f"Domain = {domain}")
        print(f"Ca = {Ca}")
        print(f"Re = {Re}")
        print(f"Ma = {Ma}")
        print(f"kb/kBT: {rbc_prms.bending_modulus() / dpd_oo_prms.kBT}")

        print(f"visc ratio = {visc_ratio}")
        print(f"dpd_outer = {dpd_oo_prms}")
        print(f"dpd_inner = {dpd_ii_prms}")
        print(f"rbc_prms = {rbc_prms}")
        print(f"mesh_ini: A={mesh_ini.area}, V={mesh_ini.volume}")
        print(f"mesh_ref: A={mesh_ref.area}, V={mesh_ref.volume}")
        print(f"length scale: {length_scale_.to('m')}")
        print(f"time scale: {time_scale_.to('s')}")
        print(f"mass scale: {mass_scale_.to('kg')}")

        sys.stdout.flush()


    return RBCShearParams(rc=rc,
                          m=mass,
                          nd=num_density,
                          RA=RA,
                          kBT=kBT,
                          L=L,
                          shear_rate=shear_rate,
                          mesh_ini=mesh_ini,
                          mesh_ref=mesh_ref,
                          rbc_params=rbc_prms,
                          dpd_oo=dpd_oo_prms,
                          dpd_ii=dpd_ii_prms,
                          dpd_io=dpd_io_prms,
                          dpd_rbco=dpd_rbco_prms,
                          dpd_rbci=dpd_rbci_prms,
                          length_scale_=length_scale_,
                          time_scale_=time_scale_,
                          mass_scale_=mass_scale_,
                          t_dump_every=0.1/shear_rate,
                          scaling_factor=scaling_factor)

def dump_parameters(p: 'RBCShearParams',
                    filename: str):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)

def load_parameters(filename: str):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_quantity(ureg: pint.UnitRegistry,
                 s: str):
    # we avoid "/" and spaces so that directory names are not weird
    if '/' in s or ' ' in s:
        raise ValueError(f"'{s}' should not contain '/' or spaces, otherwise it will create directories with undesired names. Use '_per_' and '_' instead.")
    return ureg(s.replace('_', ' '))


def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rbc-mesh-res', type=int, default=4, help='The number of subdivisions of the mesh.')
    parser.add_argument('--Re', type=float, default=0.1, help="The simulation Reynolds number.")
    parser.add_argument('--RA', type=float, default=6, help="The simulation equivalent radius of the RBC.")
    parser.add_argument('--eta-solvent', type=str, default="55.9_mPa_s", help="Viscosity of the solvent at room temperature.")
    parser.add_argument('--eta-cytosol', type=str, default="10_mPa_s", help="Viscosity of the cytosol at room temperature.")
    parser.add_argument('--shear-rate', type=str, default="10_Hz", help="Shear rate.")
    parser.add_argument('--out-params', type=str, default="parameters.pkl", help="Save all simulation parameters to this file.")
    args = parser.parse_args(argv)

    ureg = pint.UnitRegistry()

    mesh_ini = dpdprops.load_equilibrium_mesh(subdivisions=args.rbc_mesh_res)
    mesh_ref = dpdprops.load_stress_free_mesh(subdivisions=args.rbc_mesh_res)

    rbc_params = dpdprops.JuelicherLimRBCDefaultParams(ureg)

    etao_ = get_quantity(ureg, args.eta_solvent)
    etai_ = get_quantity(ureg, args.eta_cytosol)
    shear_rate_ = get_quantity(ureg, args.shear_rate)

    p = create_parameters(ureg=ureg,
                          mesh_ini=mesh_ini,
                          mesh_ref=mesh_ref,
                          rbc_params=rbc_params,
                          Re=args.Re,
                          RA=args.RA,
                          etao_=etao_,
                          etai_=etai_,
                          shear_rate_=shear_rate_,
                          verbose=True)

    dump_parameters(p, args.out_params)


if __name__ == '__main__':
    main(sys.argv[1:])
