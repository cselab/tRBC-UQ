#!/usr/bin/env python

import glob
import numpy as np
import os
import sys
import trimesh

import dpdprops

def load_mesh(fname: str):
    return trimesh.load(fname, process=False)

def color_rbc_by_rim_and_dimple(srcdir: str,
                                dstdir: str):
    if dstdir == srcdir:
        raise ValueError("The source and destination directories must be different.")

    # 1. map vertex to color accoarding to rim or dimple
    eq = dpdprops.load_equilibrium_mesh()
    nv = len(eq.vertices)
    colors = np.zeros((nv, 3)) # rgb

    x = np.array(eq.vertices[:,0])
    y = np.array(eq.vertices[:,1])
    r = np.sqrt(x**2 + y**2)
    r /= np.max(r)


    colors[:,0] = 255 # r
    colors[:,1] = 255 * r**3 # g
    colors[:,2] = 0 # b



    # 2. color the mesh and dump it
    os.makedirs(dstdir, exist_ok=True)

    ply_list = sorted(glob.glob(os.path.join(srcdir, '*.ply')))

    for fname in ply_list:
        basename = os.path.basename(fname)
        dstname = os.path.join(dstdir, basename)

        mesh = load_mesh(fname)
        mesh.visual.vertex_colors = colors

        mesh.export(dstname)



def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Color mesh from a given directory according to the original rim and dimple of the resting position.')
    parser.add_argument('srcdir', type=str, help="The directory containing the mesh to color.")
    parser.add_argument('dstdir', type=str, help="The directory that will contain the colored mesh.")
    args = parser.parse_args(argv)

    color_rbc_by_rim_and_dimple(args.srcdir, args.dstdir)


if __name__ == '__main__':
    main(sys.argv[1:])
