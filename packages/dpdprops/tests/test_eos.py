# Copyright 2020 ETH Zurich. All Rights Reserved.

import math
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import compute_pressure, sound_speed

class TestEOS(unittest.TestCase):
    def test_pressure(self):
        a = 12.5
        nd = 11
        kBT = 0.5
        rc = 1
        pref = nd * kBT + 0.103 * a * nd**2
        self.assertEqual(compute_pressure(a=a, nd=nd, kBT=kBT, rc=rc), pref)
        self.assertGreater(compute_pressure(a=11, nd=10, rc=rc, kBT=1),
                           compute_pressure(a=10, nd=10, rc=rc, kBT=1))

    def test_sound_speed(self):
        a = 12.5
        nd = 11
        kBT = 0.5
        mass = 2.0
        ref = math.sqrt(kBT / mass + 2 * 0.103 * a * nd / mass)
        self.assertEqual(sound_speed(a=a, nd=nd, kBT=kBT, rc=1, mass=mass), ref)

    def test_errors(self):
        with self.assertRaises(ValueError):
            # too small nd
            compute_pressure(a=10, nd=1, kBT=0, rc=1)

        with self.assertRaises(ValueError):
            # too small nd
            sound_speed(a=10, nd=1, kBT=0, rc=1, mass=1)

        with self.assertRaises(ValueError):
            # negative a
            sound_speed(a=-1, nd=10, kBT=0.1, rc=1, mass=1)

    def test_dimensions(self):
        from pint import UnitRegistry
        ureg = UnitRegistry()

        a = 12.5 * ureg.newton
        nd = 11 * ureg.meter**(-3)
        kBT = 1.0 * ureg.joule
        mass = 2.0 * ureg.gram
        rc = 1.0 * ureg.meter

        P = compute_pressure(a=a, nd=nd, kBT=kBT, rc=rc)
        self.assertTrue(P.check('[pressure]'))

        C = sound_speed(a=a, nd=nd, kBT=kBT, rc=rc, mass=mass)
        self.assertTrue(C.check('[velocity]'))
