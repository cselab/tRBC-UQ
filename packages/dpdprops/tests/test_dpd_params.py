# Copyright 2020 ETH Zurich. All Rights Reserved.

import math
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import \
    DPDParams, \
    create_dpd_params_from_Re_Ma, \
    create_dpd_params_from_props, \
    create_dpd_params_from_str

class TestDPDParams(unittest.TestCase):
    def test_mass_density(self):
        a = 45.0
        gamma = 15.0
        kBT = 2.0
        nd = 8.0
        mass = 1.0
        rc = 1.0
        kpow = 0.125
        p = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
        self.assertEqual(mass * nd, p.mass_density())

    def test_sigma(self):
        a = 45.0
        gamma = 15.0
        kBT = 2.0
        nd = 8.0
        mass = 1.0
        rc = 1.0
        kpow = 0.125
        p = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
        self.assertEqual(math.sqrt(2 * gamma * kBT), p.sigma())

    def test_dimensions(self):
        from pint import UnitRegistry
        ureg = UnitRegistry()

        a = 45.0 * ureg.newton
        gamma = 12.5 * ureg.gram / ureg.second
        nd = 11 * ureg.meter**(-3)
        kBT = 1.0 * ureg.joule
        rc = 1.0 * ureg.meter
        kpow = 1/2
        mass = 3 * ureg.gram

        U = 5 * ureg.meter / ureg.second
        L = 23 * ureg.meter

        p = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)

        self.assertTrue(p.self_diffusion().check('[length]**2 / [time]'))
        self.assertTrue(p.kinematic_viscosity().check('[length]**2 / [time]'))
        self.assertTrue(p.dynamic_viscosity().check('[viscosity]'))
        self.assertTrue(p.mass_density().check('[mass] / [length]**3'))
        self.assertTrue(p.sound_speed().check('[velocity]'))
        self.assertTrue(p.get_max_dt().check('[time]'))

        self.assertTrue(p.ReynoldsNumber(U, L).check('[]'))
        self.assertTrue(p.MachNumber(U).check('[]'))
        self.assertTrue(p.SchmidtNumber().check('[]'))


    def test_to_interactions(self):
        a = 45.0
        gamma = 15.0
        kBT = 2.0
        nd = 8.0
        mass = 1.0
        rc = 1.0
        kpow = 0.125
        p = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
        params = p.to_interactions()
        self.assertEqual(params['a'], a)
        self.assertEqual(params['gamma'], gamma)
        self.assertEqual(params['power'], kpow)
        self.assertEqual(params['kBT'], kBT)

    def test_to_str(self):
        a = 45.0
        gamma = 15.0
        kBT = 2.0
        nd = 8.0
        mass = 1.0
        rc = 1.0
        kpow = 0.125
        p = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
        self.assertEqual(p.to_str(), "a_45.0_gamma_15.0_kBT_2.0_nd_8.0_mass_1.0_rc_1.0_kpow_0.125")

    def test_from_str(self):
        p = create_dpd_params_from_str("a_45.0_gamma_15.0_kBT_2.0_nd_8.0_mass_1.0_rc_1.0_kpow_0.125")
        self.assertEqual(p.a, 45.0)
        self.assertEqual(p.gamma, 15.0)
        self.assertEqual(p.kBT, 2.0)
        self.assertEqual(p.nd, 8.0)
        self.assertEqual(p.mass, 1.0)
        self.assertEqual(p.rc, 1.0)
        self.assertEqual(p.kpow, 0.125)

        # with more around
        p = create_dpd_params_from_str("this_is_nothing_but_has_gamma_32_a_45.0_gamma_15.0_kBT_2.0_nd_8.0_mass_1.0_rc_1.0_kpow_0.125.annoying.extension")
        self.assertEqual(p.a, 45.0)
        self.assertEqual(p.gamma, 15.0)
        self.assertEqual(p.kBT, 2.0)
        self.assertEqual(p.nd, 8.0)
        self.assertEqual(p.mass, 1.0)
        self.assertEqual(p.rc, 1.0)
        self.assertEqual(p.kpow, 0.125)

    def test_from_to_str(self):
        s = "a_45.0_gamma_15.0_kBT_2.0_nd_8.0_mass_3.0_rc_1.12_kpow_0.125"
        p = create_dpd_params_from_str(s)
        self.assertEqual(s, p.to_str())

    def test_create_from_Re_Ma(self):
        Re = 0.1
        Ma = 0.3
        U = 1
        L = 10
        p = create_dpd_params_from_Re_Ma(Re=Re, Ma=Ma, U=U, L=L)
        self.assertAlmostEqual(p.MachNumber(U), Ma)
        self.assertAlmostEqual(p.ReynoldsNumber(U, L), Re)

    def test_create_from_props(self):
        nu = 100
        Cs = 1.5
        p = create_dpd_params_from_props(kinematic_viscosity=nu, sound_speed=Cs)
        self.assertAlmostEqual(p.kinematic_viscosity(), nu)
        self.assertAlmostEqual(p.sound_speed(), Cs)

    def test_warnings(self):
        with self.assertWarns(RuntimeWarning):
            # The optimizer will not be able to make nu that small
            nu = 1e-3
            Cs = 1.5
            create_dpd_params_from_props(kinematic_viscosity=nu, sound_speed=Cs)

    def test_timestep(self):
        a = 45.0
        gamma = 15.0
        kBT = 2.0
        nd = 8.0
        mass = 1.0
        rc = 1.0
        kpow = 0.125
        p = DPDParams(a, gamma, kBT, nd, mass, rc, kpow)
        # kind of arbitrary, testing if it runs
        self.assertLess(p.get_max_dt(), 0.1)
