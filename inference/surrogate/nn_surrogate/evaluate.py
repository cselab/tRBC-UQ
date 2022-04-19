#!/usr/bin/env python

import numpy as np
import os
import sys
import torch

from .learning import (MLP, load_model_states)

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))

class Surrogate:
    """
    Surrogate for the computational model.
    """

    def __init__(self,
                 base_dir: str):
        """
        Args:
            base_dir: The directory that contains the trained surrogates.
        """
        self.eq, self.eq_xshift, self.eq_xscale, self.eq_yshift, self.eq_yscale = load_model_states(os.path.join(base_dir, "eq.pkl"))
        self.st, self.st_xshift, self.st_xscale, self.st_yshift, self.st_yscale = load_model_states(os.path.join(base_dir, "stretch.pkl"))
        self.re, self.re_xshift, self.re_xscale, self.re_yshift, self.re_yscale = load_model_states(os.path.join(base_dir, "relax.pkl"))

        self.eq_xscale = np.array(self.eq_xscale)
        self.eq_xshift = np.array(self.eq_xshift)
        self.eq_yscale = np.array(self.eq_yscale)
        self.eq_yshift = np.array(self.eq_yshift)

        self.st_xscale = np.array(self.st_xscale)
        self.st_xshift = np.array(self.st_xshift)
        self.st_yscale = np.array(self.st_yscale)
        self.st_yshift = np.array(self.st_yshift)

        self.re_xscale = np.array(self.re_xscale)
        self.re_xshift = np.array(self.re_xshift)
        self.re_yscale = np.array(self.re_yscale)
        self.re_yshift = np.array(self.re_yshift)



    def _rescale_xeq(self, x: list):
        return (np.array(x) - self.eq_xshift) / self.eq_xscale

    def _rescale_xst(self, x: list):
        return (np.array(x) - self.st_xshift) / self.st_xscale

    def _rescale_xre(self, x: list):
        return (np.array(x) - self.re_xshift) / self.re_xscale


    def evaluate_equilibrium(self,
                             x: list):
        """
        Evaluate the surrogates related to the equilibrium shapes.

        Args:
            x: the Parameters of the model.

        Return:
            D, hmin, hmax: The diameter, min thickness and max thickness of the cell (mean and error)
        """
        assert len(x) == 4
        v, mu, FvK, b2 = x
        x = [v, FvK]
        X = torch.Tensor([self._rescale_xeq(x)])
        with torch.no_grad():
            y = self.eq(X)[0]
            y = y * self.eq_yscale + self.eq_yshift
            D, hmin, hmax = y.tolist()
        return D, hmin, hmax

    def evaluate_stretching(self,
                            x: list,
                            Fext: list):
        """
        Evaluate the surrogates related to the stretching shapes.

        Args:
            x: the Parameters of the model.
            Fext: The list of stretching forces.

        Return:
            D0, D1: The two diameters of the cell for each given stretching force (mean and error)
        """
        assert len(x) == 4
        v, mu, FvK, b2 = x

        D0 = []
        D1 = []

        for f in Fext:
            x = [v, mu, FvK, b2, f]
            X = torch.Tensor([self._rescale_xst(x)])
            with torch.no_grad():
                y = self.st(X)[0]
                y = y * self.st_yscale + self.st_yshift
            D0.append(y[0])
            D1.append(y[1])

        return D0, D1


    def evaluate_relax(self,
                       x: list):
        """
        Evaluate the surrogates related to the equilibrium shapes.

        Args:
            x: the Parameters of the model.

        Return:
            D, hmin, hmax: The diameter, min thickness and max thickness of the cell (mean and error)
        """
        assert len(x) == 5
        v, mu, FvK, b2, etam = x
        x = [v, mu, FvK, b2, etam]
        X = torch.Tensor([self._rescale_xre(x)])
        with torch.no_grad():
            y = self.re(X)[0]
            y = y * self.re_yscale + self.re_yshift
            tc = y.tolist()[0] * etam / mu
        return tc
