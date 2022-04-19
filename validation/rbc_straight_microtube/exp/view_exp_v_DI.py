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
                        default=["Tomaiuolo2009_v_DI_6.6um.csv"],
                        help="The data to plot.")
    args = parser.parse_args(argv)

    ureg = pint.UnitRegistry()

    fig, ax = plt.subplots()

    for dataset in args.csv:
        df = pd.read_csv(dataset)
        D = get_D(dataset) * ureg.um

        v = df['v'].to_numpy() * ureg.cm / ureg.s
        DI = df['DI'].to_numpy()

        if "DI_lo" in df.columns:
            dDI = (df["DI_hi"].to_numpy() - df["DI_lo"].to_numpy()) / 2
            ax.errorbar(v.to('cm/s').magnitude,
                        DI,
                        yerr=dDI,
                        fmt='o',
                        label=f"D={D}",
                        capsize=2)
        else:
            ax.plot(v.to('cm/s').magnitude,
                    DI,
                    'o',
                    label=f"D={D}")

    ax.set_xlabel(r'$v [cm/s]$')
    ax.set_ylabel(r'$DI$')

    #ax.set_xlim(0, 2)
    #ax.set_ylim(1, 3)

    ax.legend()

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
