#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import sys

def get_D(s: str):
    import re
    rexf = '[-+]?\d*\.\d+|\d+'
    matches = re.findall(f"({rexf})um", s)
    assert len(matches) == 1
    return float(matches[0])


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Plot experimental data.')
    parser.add_argument('--csv', type=str, nargs='+',
                        default=["Tomaiuolo2009_v_l_4.7um.csv", "Tomaiuolo2009_v_l_6.6um.csv"],
                        help="The data to plot.")
    args = parser.parse_args(argv)

    ureg = pint.UnitRegistry()

    fig, ax = plt.subplots()

    for dataset in args.csv:
        df = pd.read_csv(dataset)
        D = get_D(dataset) * ureg.um

        l = df['l'].to_numpy() * ureg.um
        v = df['v'].to_numpy() * ureg.cm / ureg.s

        ax.plot(v.to('cm/s').magnitude,
                l.to('um').magnitude,
                'o',
                label=f"D={D}")

    ax.set_xlabel(r'$v [cm/s]$')
    ax.set_ylabel(r'$l [um]$')

    ax.set_xlim(0.01, 10)
    ax.set_ylim(4,12)
    ax.set_xscale('log')

    ax.legend()

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
