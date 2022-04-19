# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import sys
import unittest

sys.path.insert(0, '..')
from dpdprops import (load_stress_free_mesh,
                      load_equilibrium_mesh)

class TestRbcMesh(unittest.TestCase):
    def test_load_error(self):

        with self.assertRaises(ValueError):
            load_stress_free_mesh(subdivisions=9999)

        with self.assertRaises(ValueError):
            load_equilibrium_mesh(subdivisions=9999)

    def test_load_stress_free(self):
        S0 = load_stress_free_mesh(subdivisions=3)
        nv = len(S0.vertices)
        self.assertEqual(nv, 642)

        S0 = load_stress_free_mesh(subdivisions=4)
        nv = len(S0.vertices)
        self.assertEqual(nv, 2562)


    def test_load_equilibrium(self):
        S = load_equilibrium_mesh(subdivisions=3)
        nv = len(S.vertices)
        self.assertEqual(nv, 642)

        S = load_equilibrium_mesh(subdivisions=4)
        nv = len(S.vertices)
        self.assertEqual(nv, 2562)

    def test_same_topology(self):
        for sub in [3,4]:
            S0 = load_stress_free_mesh(subdivisions=sub)
            eq = load_equilibrium_mesh(subdivisions=sub)

            faces_S0 = np.array(S0.faces)
            faces_eq = np.array(eq.faces)

            np.testing.assert_array_equal(faces_eq, faces_S0)
