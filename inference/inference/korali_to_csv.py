#!/usr/bin/env python

description = """
Extract the sample informations from korali output files
to a much more lighweight csv format
"""

import numpy as np
import pandas as pd
import sys

from utils import get_samples

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('src', type=str, help="Input json file.")
    parser.add_argument('dst', type=str, help="Output csv file.")
    args = parser.parse_args(argv)

    data = get_samples(args.src)
    pd.DataFrame(data).to_csv(args.dst, index=False)



if __name__ == '__main__':
    main(sys.argv[1:])
