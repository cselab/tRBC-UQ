#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Plot one column of csv file against another.')
    parser.add_argument('csv', type=str, help="The csv file.")
    parser.add_argument('x', type=str, help="x axis data.")
    parser.add_argument('y', type=str, help="y axis data.")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    xkey = args.x
    ykey = args.y

    x = df[xkey].to_numpy()
    y = df[ykey].to_numpy()

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
