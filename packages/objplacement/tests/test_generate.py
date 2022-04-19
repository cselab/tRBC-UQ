# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import os
import sys
import unittest

sys.path.insert(0, '..')
from objplacement import (DomainGeometry,
                          Cylinder,
                          Box,
                          generate_random_quaternions,
                          generate_positions,
                          generate_ic)

class TestGenerate(unittest.TestCase):

    def test_random_quaternions(self):
        n = 120
        qs = generate_random_quaternions(n)
        self.assertEqual(qs.shape[0], n)
        self.assertEqual(qs.shape[1], 4)

        norms = np.sum(qs**2, axis=1)
        self.assertEqual(np.mean(norms), 1)


    def test_generate_free_space(self):
        L = (12, 24, 32)
        domain = DomainGeometry(L=L)

        # put spheres inside
        R = 1
        V = 4 * np.pi / 3 * R**3
        extents = [2*R, 2*R, 2*R]
        Ht = 0.3
        pos = generate_positions(geometry=domain,
                                 obj_volume=V,
                                 obj_extents=extents,
                                 target_volume_fraction=Ht)

        self.assertEqual(len(pos.shape), 2)
        self.assertEqual(pos.shape[1], 3)

        n = len(pos)
        self.assertAlmostEqual(V * n / domain.volume(), Ht, places=3)

    def test_generate_cylinder(self):
        L = np.array((12, 24, 24))
        center = L / 2
        radius = 12
        axis = 0
        domain = Cylinder(L=L, center=center, radius=radius, axis=axis)

        # put spheres inside
        R = 0.5
        V = 4 * np.pi / 3 * R**3
        extents = [2*R, 2*R, 2*R]
        Ht = 0.3
        pos = generate_positions(geometry=domain,
                                 obj_volume=V,
                                 obj_extents=extents,
                                 target_volume_fraction=Ht)

        self.assertEqual(len(pos.shape), 2)
        self.assertEqual(pos.shape[1], 3)

        n = len(pos)
        self.assertAlmostEqual(V * n / domain.volume(), Ht, places=3)

        with self.assertRaises(RuntimeError):
            generate_positions(geometry=domain,
                               obj_volume=V,
                               obj_extents=extents,
                               target_volume_fraction=1)

    def test_generate_box(self):
        L = np.array((12, 24, 24))
        domain = Box(L=L, lo=(-10,1,1), hi=(120,24,24))

        # put spheres inside
        R = 0.5
        V = 4 * np.pi / 3 * R**3
        extents = [2*R, 2*R, 2*R]
        Ht = 0.3
        pos = generate_positions(geometry=domain,
                                 obj_volume=V,
                                 obj_extents=extents,
                                 target_volume_fraction=Ht)

        self.assertEqual(len(pos.shape), 2)
        self.assertEqual(pos.shape[1], 3)

        n = len(pos)
        self.assertAlmostEqual(V * n / domain.volume(), Ht, places=3)


    def test_generate_ic(self):
        L = (12, 24, 32)
        domain = DomainGeometry(L=L)

        # put spheres inside
        R = 1
        V = 4 * np.pi / 3 * R**3
        extents = [2*R, 2*R, 2*R]
        Ht = 0.3

        # with random quaternions

        pos_q = generate_ic(geometry=domain,
                            obj_volume=V,
                            obj_extents=extents,
                            target_volume_fraction=Ht)

        pos_q = np.array(pos_q)

        self.assertEqual(len(pos_q.shape), 2)
        self.assertEqual(pos_q.shape[1], 7)

        # with given quaternion

        with self.assertRaises(ValueError):
            # wrong orientation size
            pos_q = generate_ic(geometry=domain,
                                obj_volume=V,
                                obj_extents=extents,
                                target_volume_fraction=Ht,
                                orientation=[1,0,0])


        pos_q = generate_ic(geometry=domain,
                            obj_volume=V,
                            obj_extents=extents,
                            target_volume_fraction=Ht,
                            orientation=[1,0,0,0])

        pos_q = np.array(pos_q)

        self.assertEqual(len(pos_q.shape), 2)
        self.assertEqual(pos_q.shape[1], 7)
