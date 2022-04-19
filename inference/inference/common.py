#!/usr/bin/env python

import numpy as np

from scipy.stats import norm, truncnorm


def exponential_model(z0, zinf, tc, t):
    """
    The exponential model assumed in Hochmut 1979 for z=L/W(t)
    """
    L = (z0 + zinf) / (z0 - zinf)
    return zinf * (L + np.exp(-t/tc)) / (L - np.exp(-t/tc))


def pdf_norm(x, loc, scale):
    return norm.pdf(x, loc=loc, scale=scale)


def pdf_truncnorm(x, a, b, loc, scale):
    # bounds a and b of truncnorm are defined in the reference of the standard normal
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    a_ = (a - loc) / scale
    b_ = (b - loc) / scale
    return truncnorm.pdf(x, a=a_, b=b_, loc=loc, scale=scale)


def pdf_ratio_truncated(z,
                        ax, bx, mux, sigx,
                        ay, by, muy, sigy,
                        n: int=50):
    """
    Compute the pdf of a variable that is the ratio of
    two truncated normal distributed variables.
    """
    y = np.linspace(ay, by, n)
    dy = y[1]-y[0]

    ax_, bx_ = (ax - mux) / sigx, (bx - mux) / sigx
    ay_, by_ = (ay - muy) / sigy, (by - muy) / sigy

    pX_zy = truncnorm.pdf(np.outer(z, y), a=ax_, b=bx_, loc=mux, scale=sigx)
    pY_y = truncnorm.pdf(y, a=ay_, b=by_, loc=muy, scale=sigy)

    return np.sum(np.multiply(np.multiply(np.abs(y), pX_zy), pY_y), axis=1) * dy


def pdf_ratio_normal_truncated(z,
                               mux, sigx,
                               ay, by, muy, sigy,
                               n: int=50):
    """
    Compute the pdf of a variable that is the ratio of a normal
    distributed variable over a truncated-normal-distributed one.
    """
    ay_, by_ = (ay - muy) / sigy, (by - muy) / sigy

    y = np.linspace(ay, by, n)
    dy = y[1]-y[0]

    pX_zy = norm.pdf(np.outer(z, y), loc=mux, scale=sigx)
    pY_y = truncnorm.pdf(y, a=ay_, b=by_, loc=muy, scale=sigy)

    return np.sum(np.multiply(np.multiply(np.abs(y), pX_zy), pY_y), axis=1) * dy
