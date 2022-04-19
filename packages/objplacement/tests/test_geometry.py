# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np
import os
import sys
import unittest

sys.path.insert(0, '..')
from objplacement import (DomainGeometry,
                          Cylinder,
                          Box,
                          FromSdfTools)

class TestGeometery(unittest.TestCase):

    def test_domain_creation_checks(self):

        with self.assertRaises(ValueError):
            c = DomainGeometry(L=(1,2))

        with self.assertRaises(ValueError):
            c = DomainGeometry(L=(-1,2,2))

        c = DomainGeometry(L=(1,2,3))

    def test_domain_inside(self):
        L = (10, 20, 30)
        domain = DomainGeometry(L=L)

        positions = np.array([[-1,2,3],
                              [5,2,3],
                              [1,8,3],
                              [11,8,8],
                              [1,22,33]])
        ref = np.array([False, True, True, False, False])

        np.testing.assert_equal(domain.points_inside(positions), ref)

        with self.assertRaises(ValueError):
            domain.points_inside(np.array([[0,1]]))

    def test_domain_volume(self):
        L = (10, 20, 30)
        domain = DomainGeometry(L=L)
        self.assertEqual(domain.volume(), L[0] * L[1] * L[2])



    def test_cylinder_creation_checks(self):

        # wrong center
        with self.assertRaises(ValueError):
            c = Cylinder(L=(10,10,10), radius=1, center=(0,1), axis=0)

        # wrong axis
        with self.assertRaises(ValueError):
            c = Cylinder(L=(10,10,10), radius=1, center=(0,0,0), axis=4)

        # cylinder does not fit in the domain
        with self.assertRaises(ValueError):
            c = Cylinder(L=(10000,10,10), radius=100, center=(5,5,5), axis=1)

    def test_cylinder_inside(self):
        L      = (32, 32, 32)
        R      = 5
        center = (10,20,15)
        axis   = 0
        cyl    = Cylinder(L=L, radius=R, center=center, axis=axis)

        positions = np.array([[1,20,18],
                              [5,20,18],
                              [5,11,15]])
        ref = np.array([True, True, False])

        np.testing.assert_equal(cyl.points_inside(positions), ref)

        with self.assertRaises(ValueError):
            cyl.points_inside(np.array([[0,1]]))

    def test_cylinder_volume(self):
        L      = (32, 32, 32)
        R      = 5
        center = (10,20,15)
        axis   = 0
        cyl    = Cylinder(L=L, radius=R, center=center, axis=axis)

        self.assertEqual(cyl.volume(), L[0] * np.pi * R**2)


    def test_box_creation_checks(self):
        # wrong dims
        with self.assertRaises(ValueError):
            b = Box(L=(12, 12, 12), lo=(1,2), hi=(12,2,3,4))

        # lo higher than hi
        with self.assertRaises(ValueError):
            b = Box(L=(12, 12, 12), lo=(1000, 2, 2), hi=(5, 5, 5))


    def test_box_inside(self):
        L = (10, 20, 30)
        lo = (-10, 1, 1)
        hi = (120, 19, 29)
        domain = Box(L=L, lo=lo, hi=hi)

        positions = np.array([[-1,2,3],
                              [5,2,3],
                              [1,8,3],
                              [11,8,8],
                              [1,22,33]])
        ref = np.array([False, True, True, False, False])

        np.testing.assert_equal(domain.points_inside(positions), ref)

        with self.assertRaises(ValueError):
            domain.points_inside(np.array([[0,1]]))


    def test_box_volume(self):

        b1 = Box(L=(12, 12, 12), lo=(-10, 0, 0), hi=(120, 12, 12))
        self.assertEqual(b1.volume(), 12**3)

        b2 = Box(L=(12, 12, 12), lo=(1, 1, 1), hi=(2, 2, 2))
        self.assertEqual(b2.volume(), 1)


    def test_sdf_tools(self):
        import sdf_tools.Sdf as sdf
        L = (2, 2, 2)
        R = 0.5
        my_sdf = sdf.Sphere(center=(1, 1, 1), radius=R, inside=True)

        geom = FromSdfTools(L, my_sdf)
        exact_volume = np.pi * R**3 * 4 / 3

        self.assertAlmostEqual(geom.volume(), exact_volume, places=2)
