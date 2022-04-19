#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd

def plot(files: list,
         colors: list):
    fig, ax = plt.subplots()
    for filename, color in zip(files, colors):
        df = pd.read_csv(filename)
        Fext = df['Fext'].to_numpy()
        D0 = df['D0'].to_numpy()
        D1 = df['D1'].to_numpy()

        ax.plot(Fext, D0, '-o', color=color)
        ax.plot(Fext, D1, '-s', color=color)

    ax.set_xlabel(r'$F_{ext}$[pN]')
    ax.set_ylabel(r'$D$[$\mu$m]')
    ax.set_xlim(0,)

    plt.savefig('data.pdf', transparent=True)


if __name__ == '__main__':
    plot(['mills_2004/mills_2004.csv',
          'suresh_2005/suresh_2005.csv'],
         ['k', 'darkred'])
