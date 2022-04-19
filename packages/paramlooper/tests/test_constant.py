# Copyright 2020 ETH Zurich. All Rights Reserved.

import sys
import unittest

sys.path.insert(0, '..')
from paramlooper import ParamConst

class TestConstant(unittest.TestCase):
    def test_iter(self):
        p = ParamConst(42)
        s = sum(val for val in p)
        self.assertEqual(s, 42)
