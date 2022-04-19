# Copyright 2020 ETH Zurich. All Rights Reserved.

import sys
import unittest

sys.path.insert(0, '..')
from paramlooper import (ParamList,
                         ParamLinear,
                         ParamLog)

class TestLists(unittest.TestCase):
    def test_iter_list(self):
        mylist = [42, 43, 44]
        params = ParamList(mylist)
        plist = [val for val in params]

        for l0, lp in zip(mylist, plist):
            self.assertEqual(l0, lp)


    def test_iter_linear(self):
        params = ParamLinear(0, 5, 6)

        for i, val in enumerate(params):
            self.assertEqual(float(i), val)


    def test_iter_log(self):
        n = 6
        params = ParamLog(1, 10**n, n+1)

        for i, val in enumerate(params):
            self.assertAlmostEqual(float(10**i), val)

    def test_exceptions_log(self):
        with self.assertRaises(ValueError):
            params = ParamLog(0, 10, 3)
        with self.assertRaises(ValueError):
            params = ParamLog(10, 0, 3)
