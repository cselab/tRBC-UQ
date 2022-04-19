# Copyright 2020 ETH Zurich. All Rights Reserved.

import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import (kantor_to_bending_modulus,
                      bending_modulus_to_kantor)

class TestKantor(unittest.TestCase):
    def test_dimensions(self):
        from pint import UnitRegistry
        ureg = UnitRegistry()

        kc = 100 * ureg.mN / ureg.m * (50 * ureg.nm)**2

        kb = bending_modulus_to_kantor(kc=kc)
        self.assertTrue(kb.check('[energy]'))

    def test_inverse(self):
        kc_ref = 100
        kb = bending_modulus_to_kantor(kc=kc_ref)
        kc = kantor_to_bending_modulus(kb=kb)
        self.assertEqual(kc, kc_ref)
