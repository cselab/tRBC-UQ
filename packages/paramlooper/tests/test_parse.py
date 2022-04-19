# Copyright 2020 ETH Zurich. All Rights Reserved.

import sys
import unittest

sys.path.insert(0, '..')
from paramlooper import (make_param,
                         ParamConst,
                         ParamList,
                         ParamLinear,
                         ParamLog)

class TestParse(unittest.TestCase):
    def test_const(self):
        for r, v in zip([42.0], make_param("42")):
            self.assertEqual(r, v)

        for r, v in zip([42.0], make_param("const,42")):
            self.assertEqual(r, v)

        with self.assertRaises(SyntaxError):
            make_param("const,42,43")

    def test_list(self):
        for r, v in zip([42.0, 43.0], make_param("list,42,43")):
            self.assertEqual(r, v)

        with self.assertRaises(SyntaxError):
            make_param("list")

    def test_linear(self):
        for r, v in zip([1.0, 2.0, 3.0, 4.0], make_param("linear,1,4,4")):
            self.assertEqual(r, v)

        with self.assertRaises(SyntaxError):
            make_param("linear,1,4")

    def test_log(self):
        for r, v in zip([1.0, 10.0, 100.0, 1000.0], make_param("log,1,1000,4")):
            self.assertAlmostEqual(r, v)

        with self.assertRaises(SyntaxError):
            make_param("log,1,4")

    def test_unknown(self):
        with self.assertRaises(SyntaxError):
            make_param("unknown,12")
