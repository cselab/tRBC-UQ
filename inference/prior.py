#!/usr/bin/env python

from dataclasses import dataclass
import numpy as np

RA = np.sqrt(135/(4*np.pi))

class Prior:
    def __init__(self):
        pass

    def pdf(self, x):
        raise NotImplementedError("Not implemented.")

    def logpdf(self, x):
        raise NotImplementedError("Not implemented.")

    def low(self):
        return -np.inf

    def high(self):
        return +np.inf

    def configure_korali(self, kexp, i):
        raise NotImplementedError("Not implemented.")


class UniformPrior(Prior):
    """
    Uniformly distributed prior in [a, b].
    """
    def __init__(self, a, b):
        if a >= b:
            raise ValueError("Wrong uniform prior bounds, must get a < b.")
        super().__init__()
        self.a = a
        self.b = b

    def pdf(self, x):
        return np.where(np.logical_and(x >= self.a, x <= self.b),
                        1.0 / (self.b - self.a),
                        0.0)

    def logpdf(self, x):
        return np.where(np.logical_and(x >= self.a, x <= self.b),
                        -np.log(self.b - self.a),
                        -np.inf)

    def low(self):
        return self.a

    def high(self):
        return self.b

    def configure_korali(self, e, i):
            e["Distributions"][i]["Type"] = "Univariate/Uniform"
            e["Distributions"][i]["Minimum"] = self.a
            e["Distributions"][i]["Maximum"] = self.b


class RatioUniformPrior(Prior):
    """
    Ratio of variables uniformly distributed, Z = X/Y with
    X ~ U(a,b) and Y ~ U(c, d)
    """
    def __init__(self, a, b, c, d):
        if a >= b or a <= 0:
            raise ValueError("Wrong prior bounds, must get 0 < a < b.")
        if c >= d or c <= 0:
            raise ValueError("Wrong prior bounds, must get 0 < c < d.")
        super().__init__()

        b0 = a/d
        b1 = a/c
        b2 = b/d
        b3 = b/c

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.b0 = b0
        self.b3 = b3
        self.norm = d**2 * (b1 - b0) + a**2 * (1/b1 - 1/b0) + \
            (d**2 - c**2) * (b2 - b1) + \
            -b**2 * (1/b3 - 1/b2) - c**2 * (b3 - b2)


    def pdf(self, x):
        return np.where(np.logical_and(x >= self.b0, x <= self.b3),
                        (np.minimum(self.d, self.b/x)**2 - np.maximum(self.c, self.a/x)**2) / self.norm,
                        0.0)

    def logpdf(self, x):
        return np.where(np.logical_and(x >= self.b0, x <= self.b3),
                        np.log(np.minimum(self.d, self.b/x)**2 - np.maximum(self.c, self.a/x)**2) - np.log(self.norm),
                        -np.inf)

    def low(self):
        return self.b0

    def high(self):
        return self.b3

    def configure_korali(self, e, i):
        e["Distributions"][i]["Type"] = "Univariate/UniformRatio"
        e["Distributions"][i]["Minimum X"] = self.a
        e["Distributions"][i]["Maximum X"] = self.b
        e["Distributions"][i]["Minimum Y"] = self.c
        e["Distributions"][i]["Maximum Y"] = self.d


@dataclass
class VarConfig:
    name: str
    prior: Prior

    def low(self):
        return self.prior.low()

    def high(self):
        return self.prior.high()

    def configure_korali(self, e, i):
        prior_name = f"Prior {self.name}"
        e["Distributions"][i]["Name"] = prior_name
        e["Variables"][i]["Name"] = self.name
        e["Variables"][i]["Prior Distribution"] = prior_name
        self.prior.configure_korali(e, i)


def _to_dict(vars: list):
    d = dict()
    for var in vars:
        d[var.name] = var
    return d


conf_kb = VarConfig(name="kb",
                    prior=UniformPrior(a=0.1074295866, # 1e-18 J
                                       b=1.074295866))

comp_variables = [
    VarConfig(name="v",
              prior=UniformPrior(a=0.65, b=1.00)),

    VarConfig(name="mu",
              prior=UniformPrior(a=1.0, b=10.0)), # uN/m

    conf_kb,

    VarConfig(name="FvK",
              prior=RatioUniformPrior(a=1.0, b=10.0, # mu, uN/m
                                      c=conf_kb.low()/RA**2, d=conf_kb.high()/RA**2)), # kb / RA**2, uN/m

    VarConfig(name="b2",
              prior=UniformPrior(a=0.0, b=4.0)),

    VarConfig(name="etam",
              prior=UniformPrior(a=0.1, b=1.0)) # Pa.s.um
]

input_variables = [
    VarConfig(name="Fext",
              prior=UniformPrior(a=0.0, b=200.0)) # pN
]

comp_variables_dict = _to_dict(comp_variables)
input_variables_dict = _to_dict(input_variables)


surrogate_variables = [
    comp_variables_dict["v"],
    comp_variables_dict["mu"],
    comp_variables_dict["FvK"],
    comp_variables_dict["b2"],
    comp_variables_dict["etam"],
    *input_variables
]

surrogate_variables_dict = _to_dict(surrogate_variables)




if __name__ == '__main__':

    def print_var_list(vars):
        for var in vars:
            print(f"\t{var.name}:\t [{var.low()}, {var.high()}]")


    print("Computational variables:")
    print_var_list(comp_variables)
    print()

    print("Input variables:")
    print_var_list(input_variables)
    print()

    print("Surrogate variables:")
    print_var_list(surrogate_variables)
    print()

    FvK = comp_variables_dict['FvK']
    print(f"kb in [{10*FvK.prior.c*RA**2}, {10*FvK.prior.d*RA**2}] X 1e-19 J")

    # plot the priors

    def plot_var(var, ax):
        L = var.high() - var.low()
        val = np.linspace(var.low() - L/10, var.high() + L/10, 1000)
        pdf = var.prior.pdf(val)
        ax.plot(val, pdf)
        ax.set_xlabel(var.name)

    import matplotlib.pyplot as plt
    fix, axes = plt.subplots(2, 2)
    plot_var(comp_variables_dict['v'], axes[0,0])
    plot_var(comp_variables_dict['mu'], axes[1,0])
    plot_var(comp_variables_dict['FvK'], axes[0,1])
    plot_var(comp_variables_dict['b2'], axes[1,1])
    plt.tight_layout()
    plt.show()
