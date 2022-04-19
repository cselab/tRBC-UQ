#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd

def plot(files: list,
         markers: list):

    fig, ax = plt.subplots()

    for filename, marker in zip(files, markers):
        df = pd.read_csv(filename)
        t = df['t'].to_numpy()
        L_W = df['L_W'].to_numpy()

        ax.plot(t, L_W, 'k', marker=marker)

    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$D_{ax}/D_{tr}$')
    ax.set_xlim(0,)

    #plt.show()
    plt.savefig('data.pdf', transparent=True)


if __name__ == '__main__':
    plot([f'hochmut_1979/{i}.csv' for i in range(1,5)],
         markers=['o', 's', '+', '<'])
