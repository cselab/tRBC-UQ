# Copyright 2020 ETH Zurich. All Rights Reserved.

import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import (shear_modulus_to_wlc,
                      wlc_to_shear_modulus)

class TestWlc(unittest.TestCase):
    def test_dimensions(self):
        from pint import UnitRegistry
        ureg = UnitRegistry()

        m = 2
        mu = 100 * ureg.mN / ureg.m

        l0 = 0.1 * ureg.um
        x0 = 0.3

        ks = shear_modulus_to_wlc(mu=mu, l0=l0, x0=x0, m=m)
        self.assertTrue(ks.check('[force]'))
        self.assertTrue(wlc_to_shear_modulus(ks=ks, l0=l0, x0=x0, m=m).check('[force] / [length]'))

    def test_inverse(self):
        m = 2
        mu_ref = 123

        l0 = 0.1
        x0 = 0.3

        ks = shear_modulus_to_wlc(mu=mu_ref, l0=l0, x0=x0, m=m)
        mu = wlc_to_shear_modulus(ks=ks, l0=l0, x0=x0, m=m)
        self.assertAlmostEqual(mu_ref, mu)
