# Copyright 2020 ETH Zurich. All Rights Reserved.

from .param import (ParamConst,
                    ParamList,
                    ParamLinear,
                    ParamLog)

def _isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def make_param(s: str):
    """
    Create a parameter class from string.
    """
    l = s.split(',')

    if len(l) == 1 and _isfloat(s):
        return ParamConst(float(s))

    elif l[0] == 'const':
        if len(l) != 2:
            raise SyntaxError("ParamConst must have one float, syntax: 'const,42'.")
        return ParamConst(float(l[1]))

    elif l[0] == 'list':
        if len(l) < 2:
            raise SyntaxError("ParamList must have at least one float, syntax: 'list,42,43,44'.")
        return ParamList([float(v) for v in l[1:]])

    elif l[0] == 'linear':
        if len(l) != 4:
            raise SyntaxError("ParamLinear must have 3 parameters, syntax: 'linear,0.0,10.0,5'")
        return ParamLinear(float(l[1]), float(l[2]), int(l[3]))

    elif l[0] == 'log':
        if len(l) != 4:
            raise SyntaxError("ParamLog must have 3 parameters, syntax: 'log,0.0,10.0,5'")
        return ParamLog(float(l[1]), float(l[2]), int(l[3]))

    else:
        raise SyntaxError(f"Wrong format: {s}")
