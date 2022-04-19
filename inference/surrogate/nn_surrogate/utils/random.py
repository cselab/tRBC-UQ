#!/usr/bin/env python

import numpy as np

class TemporaryNumpySeed:
    """
    see https://stackoverflow.com/a/49557099/11630848
    """
    def __init__(self, seed):
        self.seed = seed
        self.backup = None

    def __enter__(self):
        self.backup = np.random.randint(2**32-1, dtype=np.uint32)
        np.random.seed(self.seed)

    def __exit__(self, *_):
        np.random.seed(self.backup)
