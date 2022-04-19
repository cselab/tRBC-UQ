# Copyright 2020 ETH Zurich. All Rights Reserved.

import numpy as np

class ParamConst:
    """
    Wrapper for a single value parameter.
    Can be iterated, will yield its only value.
    """

    def __init__(self, val):
        """
        Arguments:
            val: The value that should be wrapped.
        """
        self.val = val

    def __iter__(self):
        """ Iterate over the (single) value of the Param object."""
        yield self.val


class ParamList:
    """
    Wrapper for a list of values.
    Can be iterated, will loop in the given list order.
    """

    def __init__(self, values: list):
        """
        Arguments:
            values: The list of values that should be wrapped.
        """
        self.vals = values

    def __iter__(self):
        """ Iterate over the all values of the object."""
        for val in self.vals:
            yield val


class ParamLinear(ParamList):
    """
    List of floats equally spaced on an interval.
    See numpy linspace function.
    """

    def __init__(self,
                 start: float,
                 end: float,
                 n: int):
        """
        Arguments:
            start: First value of the sequence (inclusive)
            end: Last value of the sequence (inclusive)
            n: number of values in the sequence.
        """
        super().__init__(np.linspace(start, end, n).tolist())


class ParamLog(ParamList):
    """
    List of floats equally spaced on an interval in logarithmic scale.
    See numpy logspace function.
    """

    def __init__(self,
                 start: float,
                 end: float,
                 n: int):
        """
        Arguments:
            start: First value of the sequence (inclusive). Must be positive.
            end: Last value of the sequence (inclusive). Must be positive.
            n: number of values in the sequence.
        """
        if start <= 0:
            raise ValueError(f"start must be positive, got {start}.")
        if end <= 0:
            raise ValueError(f"end must be positive, got {end}.")

        seq = np.logspace(np.log(start) / np.log(10),
                          np.log(end) / np.log(10),
                          n)
        super().__init__(seq.tolist())
